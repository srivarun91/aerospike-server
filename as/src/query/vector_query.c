/*
 * vector_query.c
 *
 * Copyright (C) 2022 - 2025 Aerospike, Inc.
 *
 * Portions may be licensed to Aerospike, Inc. under one or more contributor
 * license agreements.
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/
 */

//==============================================================================
// Includes.
//

#include "query/vector_query.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "citrusleaf/alloc.h"
#include "citrusleaf/cf_byte_order.h"
#include "citrusleaf/cf_digest.h"

#include "dynbuf.h"
#include "log.h"
#include "cf_mutex.h"
#include "socket.h"

#include "base/datamodel.h"
#include "base/index.h"
#include "base/proto.h"
#include "base/transaction.h"
#include "storage/storage.h"
#include "query/query_job.h"
#include "query/query_manager.h"


//==============================================================================
// Vector distance query implementation.
//

//==============================================================================
// Vector serialization format definition
//

#define VECTOR_MAGIC 0x56454354 // "VECT"
#define VECTOR_VERSION 1

typedef enum {
	VECTOR_TYPE_FLOAT32 = 1,
	VECTOR_TYPE_FLOAT64 = 2,
	VECTOR_TYPE_INT32 = 3,
	VECTOR_TYPE_INT64 = 4
} vector_element_type;

typedef struct vector_header_s {
	uint32_t magic;        // Magic number for validation
	uint32_t version;      // Format version
	uint32_t element_count; // Number of elements in vector
	uint32_t element_type;  // Type of elements (vector_element_type)
} __attribute__((packed)) vector_header;

#define VECTOR_HEADER_SIZE sizeof(vector_header)

// Connection query job structure (copied from query.c)
typedef struct conn_query_job_s {
	as_query_job _base;

	cf_mutex fd_lock;
	as_file_handle* fd_h;
	int32_t fd_timeout;
	bool compress_response;
	uint64_t net_io_bytes;
	uint64_t net_io_ns;
} conn_query_job;

// Vector distance query job structure
typedef struct vector_distance_query_job_s {
	conn_query_job _base;

	uint8_t* query_vector_blob;
	uint32_t query_vector_blob_size;
	vector_header query_vector_header;
	const uint8_t* query_vector_data;
} vector_distance_query_job;

// Forward declarations for vector distance query
static void vector_distance_query_job_slice(as_query_job* _job, as_partition_reservation* rsv, cf_buf_builder** bb_r);
static void vector_distance_query_job_finish(as_query_job* _job);
static void vector_distance_query_job_destroy(as_query_job* _job);
static void vector_distance_query_job_info(as_query_job* _job, cf_dyn_buf* db);
static bool vector_distance_query_record_matches(as_index_ref* r_ref, void* udata);

// Connection job helper functions
static void conn_query_job_init(conn_query_job* job, const as_transaction* tr);
static void conn_query_job_destroy(conn_query_job* job);
static void conn_query_job_finish(conn_query_job* job);
static bool conn_query_job_send_response(conn_query_job* job, uint8_t* buf, size_t size);
static void conn_query_job_release_fd(conn_query_job* job, bool force_close);
static size_t send_blocking_response_chunk(as_file_handle* fd_h, uint8_t* buf, size_t size, int32_t timeout, bool compress, as_proto_comp_stat* comp_stat);

// Vector distance query job vtable
static const as_query_vtable vector_distance_query_job_vtable = {
	vector_distance_query_job_slice,
	vector_distance_query_job_finish,
	vector_distance_query_job_destroy,
	vector_distance_query_job_info
};

// Parse vector from blob data
static bool
parse_vector_blob(const uint8_t* blob_data, uint32_t blob_size,
		vector_header* header, const uint8_t** vector_data)
{
	if (blob_size < VECTOR_HEADER_SIZE) {
		cf_warning(AS_QUERY, "vector blob too small for header: %u bytes", blob_size);
		return false;
	}

	// Copy and validate header
	memcpy(header, blob_data, VECTOR_HEADER_SIZE);

	// Convert from network byte order if needed
	header->magic = cf_swap_from_be32(header->magic);
	header->version = cf_swap_from_be32(header->version);
	header->element_count = cf_swap_from_be32(header->element_count);
	header->element_type = cf_swap_from_be32(header->element_type);

	// Validate magic number
	if (header->magic != VECTOR_MAGIC) {
		cf_warning(AS_QUERY, "invalid vector magic: 0x%08x", header->magic);
		return false;
	}

	// Validate version
	if (header->version != VECTOR_VERSION) {
		cf_warning(AS_QUERY, "unsupported vector version: %u", header->version);
		return false;
	}

	// Validate element type
	if (header->element_type < VECTOR_TYPE_FLOAT32 || header->element_type > VECTOR_TYPE_INT64) {
		cf_warning(AS_QUERY, "invalid vector element type: %u", header->element_type);
		return false;
	}

	// Calculate expected data size
	uint32_t element_size;
	switch (header->element_type) {
		case VECTOR_TYPE_FLOAT32:
		case VECTOR_TYPE_INT32:
			element_size = 4;
			break;
		case VECTOR_TYPE_FLOAT64:
		case VECTOR_TYPE_INT64:
			element_size = 8;
			break;
		default:
			return false;
	}

	uint32_t expected_data_size = header->element_count * element_size;
	uint32_t total_expected_size = VECTOR_HEADER_SIZE + expected_data_size;

	if (blob_size != total_expected_size) {
		cf_warning(AS_QUERY, "vector blob size mismatch: got %u, expected %u",
			blob_size, total_expected_size);
		return false;
	}

	// Set vector data pointer
	*vector_data = blob_data + VECTOR_HEADER_SIZE;

	return true;
}

// Calculate Euclidean distance between two vectors
static double
calculate_vector_distance(const vector_header* header1, const uint8_t* data1,
		const vector_header* header2, const uint8_t* data2)
{
	// Vectors must have same element count and type
	if (header1->element_count != header2->element_count) {
		cf_warning(AS_QUERY, "vector dimension mismatch: %u vs %u",
			header1->element_count, header2->element_count);
		return -1.0; // Return negative to indicate error
	}

	if (header1->element_type != header2->element_type) {
		cf_warning(AS_QUERY, "vector type mismatch: %u vs %u",
			header1->element_type, header2->element_type);
		return -1.0; // Return negative to indicate error
	}

	double sum_squared_diff = 0.0;
	uint32_t count = header1->element_count;

	switch (header1->element_type) {
		case VECTOR_TYPE_FLOAT32: {
			const float* vec1 = (const float*)data1;
			const float* vec2 = (const float*)data2;
			for (uint32_t i = 0; i < count; i++) {
				double diff = (double)vec1[i] - (double)vec2[i];
				sum_squared_diff += diff * diff;
			}
			break;
		}
		case VECTOR_TYPE_FLOAT64: {
			const double* vec1 = (const double*)data1;
			const double* vec2 = (const double*)data2;
			for (uint32_t i = 0; i < count; i++) {
				double diff = vec1[i] - vec2[i];
				sum_squared_diff += diff * diff;
			}
			break;
		}
		case VECTOR_TYPE_INT32: {
			const int32_t* vec1 = (const int32_t*)data1;
			const int32_t* vec2 = (const int32_t*)data2;
			for (uint32_t i = 0; i < count; i++) {
				double diff = (double)vec1[i] - (double)vec2[i];
				sum_squared_diff += diff * diff;
			}
			break;
		}
		case VECTOR_TYPE_INT64: {
			const int64_t* vec1 = (const int64_t*)data1;
			const int64_t* vec2 = (const int64_t*)data2;
			for (uint32_t i = 0; i < count; i++) {
				double diff = (double)vec1[i] - (double)vec2[i];
				sum_squared_diff += diff * diff;
			}
			break;
		}
		default:
			cf_warning(AS_QUERY, "unsupported vector type for distance calculation: %u",
				header1->element_type);
			return -1.0; // Return negative to indicate error
	}

	// Return Euclidean distance (square root of sum of squared differences)
	return sqrt(sum_squared_diff);
}

// Get vector data from transaction and parse it
static bool
get_query_vector(const as_transaction* tr, vector_distance_query_job* job)
{
	const as_msg_field* f = as_msg_field_get(&tr->msgp->msg, AS_MSG_FIELD_TYPE_VECTOR_OP);

	if (f == NULL) {
		cf_warning(AS_QUERY, "missing vector-op field");
		return false;
	}

	job->query_vector_blob_size = as_msg_field_get_value_sz(f);

	if (job->query_vector_blob_size == 0) {
		cf_warning(AS_QUERY, "empty vector data");
		return false;
	}

	// Copy the vector blob data
	job->query_vector_blob = cf_malloc(job->query_vector_blob_size);
	memcpy(job->query_vector_blob, f->data, job->query_vector_blob_size);

	// Parse the vector blob
	if (!parse_vector_blob(job->query_vector_blob, job->query_vector_blob_size,
			&job->query_vector_header, &job->query_vector_data)) {
		cf_warning(AS_QUERY, "failed to parse query vector");
		return false;
	}

	cf_info(AS_QUERY, "parsed query vector: %u elements of type %u",
		job->query_vector_header.element_count, job->query_vector_header.element_type);

	return true;
}

// Vector distance query job start function
int
vector_distance_query_job_start(as_transaction* tr, as_namespace* ns)
{
	vector_distance_query_job* job = cf_calloc(1, sizeof(vector_distance_query_job));
	conn_query_job* conn_job = (conn_query_job*)job;
	as_query_job* _job = (as_query_job*)job;

	// Initialize base query job with vtable
	as_query_job_init(_job, &vector_distance_query_job_vtable, tr, ns);

	// Take ownership of socket from transaction
	conn_query_job_init(conn_job, tr);

	// Get vector data from transaction
	if (! get_query_vector(tr, job)) {
		conn_query_job_destroy(conn_job);
		as_query_job_destroy(_job);
		return AS_ERR_PARAMETER;
	}

    // TODO(varun): make it a detail/debug log.
	cf_info(AS_QUERY, "starting vector distance query job %lu", _job->trid);

	// Start the query job using query manager
	int result = as_query_manager_start_job(_job);

	if (result != AS_OK) {
		conn_query_job_destroy(conn_job);
		as_query_job_destroy(_job);
	}

	return result;
}

// Vector distance query job slice - processes one partition
static void
vector_distance_query_job_slice(as_query_job* _job, as_partition_reservation* rsv,
        cf_buf_builder** bb_r)
{
	vector_distance_query_job* job = (vector_distance_query_job*)_job;

	cf_info(AS_QUERY, "vector distance query job %lu processing partition %u",
		_job->trid, rsv->p->id);

	// Use the provided buffer builder
	if (*bb_r == NULL) {
		*bb_r = cf_buf_builder_create_size(1024 * 1024); // 1MB
	}

	// Create context for record matching
	vector_record_context ctx = {
		.job = job,
		.bb = *bb_r
	};

	// Scan all records in this partition
	as_index_reduce(rsv->tree, vector_distance_query_record_matches, &ctx);

	// Send accumulated results to client if buffer has data
	cf_buf_builder* bb = *bb_r;
	if (bb->used_sz > sizeof(as_proto)) {
		conn_query_job_send_response((conn_query_job*)job, bb->buf, bb->used_sz);
		cf_buf_builder_reset(bb);
	}
}

// Context structure to pass buffer builder to record matching function
typedef struct vector_record_context_s {
	vector_distance_query_job* job;
	cf_buf_builder* bb;
} vector_record_context;

// Process individual record for vector distance calculation
static bool
vector_distance_query_record_matches(as_index_ref* r_ref, void* udata)
{
	vector_record_context* ctx = (vector_record_context*)udata;
	vector_distance_query_job* job = ctx->job;
	as_query_job* _job = (as_query_job*)job;
	as_namespace* ns = _job->ns;
	as_record* r = r_ref->r;

	// Check if record is valid and live
	if (! as_record_is_live(r)) {
		as_record_done(r_ref, ns);
		return true;
	}

	// Check set filter (assuming excluded_set function exists)
	if (_job->set_id != INVALID_SET_ID && as_index_get_set_id(r) != _job->set_id) {
		as_record_done(r_ref, ns);
		return true;
	}

	// Read record data
	as_storage_rd rd;
	as_storage_record_open(ns, r, &rd);

	// Look for vector bins in the record
	for (uint16_t i = 0; i < rd.n_bins; i++) {
		as_bin* b = &rd.bins[i];

		if (b->particle != NULL && b->particle->type == AS_PARTICLE_TYPE_VECTOR) {
			// Get the blob data from the particle
			char* record_vector_blob;
			uint32_t blob_size = as_bin_particle_string_ptr(b, &record_vector_blob);

			if (record_vector_blob == NULL || blob_size == 0) {
				continue; // Skip invalid vector data
			}

			// Parse the record's vector
			vector_header record_header;
			const uint8_t* record_vector_data;

			if (!parse_vector_blob((const uint8_t*)record_vector_blob, blob_size,
					&record_header, &record_vector_data)) {
				cf_debug(AS_QUERY, "failed to parse record vector for %pD", &r->keyd);
				continue; // Skip invalid vector
			}

			// Calculate distance
			double distance = calculate_vector_distance(
				&job->query_vector_header, job->query_vector_data,
				&record_header, record_vector_data);

			// Skip records with invalid distance (negative indicates error)
			if (distance < 0.0) {
				continue;
			}

			// Add result to response buffer
			cf_buf_builder_append_uint64(ctx->bb, (uint64_t)r->keyd.digest[0]);
			cf_buf_builder_append_uint64(ctx->bb, (uint64_t)r->keyd.digest[1]);
			cf_buf_builder_append_uint64(ctx->bb, *(uint64_t*)&distance); // Store as uint64

			as_incr_uint64(&_job->n_succeeded);

			cf_debug(AS_QUERY, "vector distance calculated: %f for record %pD",
				distance, &r->keyd);
		}
	}

	as_storage_record_close(&rd);
	as_record_done(r_ref, ns);

	return true;
}

// Finish vector distance query job
static void
vector_distance_query_job_finish(as_query_job* _job)
{
	vector_distance_query_job* job = (vector_distance_query_job*)_job;

	cf_info(AS_QUERY, "finishing vector distance query job %lu", _job->trid);

	// Use connection job finish to handle client communication
	conn_query_job_finish((conn_query_job*)job);
}

// Destroy vector distance query job
static void
vector_distance_query_job_destroy(as_query_job* _job)
{
	vector_distance_query_job* job = (vector_distance_query_job*)_job;

	if (job->query_vector_blob != NULL) {
		cf_free(job->query_vector_blob);
	}

	// Note: bb is managed by the query framework, not by us
}

// Info for vector distance query job
static void
vector_distance_query_job_info(as_query_job* _job, cf_dyn_buf* db)
{
	vector_distance_query_job* job = (vector_distance_query_job*)_job;

	cf_dyn_buf_append_string(db, "vector-distance-query:");
	cf_dyn_buf_append_uint64(db, _job->trid);
	cf_dyn_buf_append_string(db, ":vector-elements=");
	cf_dyn_buf_append_uint32(db, job->query_vector_header.element_count);
	cf_dyn_buf_append_string(db, ":vector-type=");
	cf_dyn_buf_append_uint32(db, job->query_vector_header.element_type);
}

//==============================================================================
// Connection job helper implementations (copied from query.c)
//

static void
conn_query_job_init(conn_query_job* job, const as_transaction* tr)
{
	cf_mutex_init(&job->fd_lock);

	job->fd_h = tr->from.proto_fd_h;
	as_file_handle_reserve(job->fd_h);
}

static void
conn_query_job_destroy(conn_query_job* job)
{
	cf_mutex_destroy(&job->fd_lock);
}

static void
conn_query_job_finish(conn_query_job* job)
{
	as_query_job* _job = (as_query_job*)job;

	if (job->fd_h) {
		if (_job->is_short) {
			conn_query_job_release_fd(job, false);
		}
		else {
			uint64_t before_ns = cf_getns();
			size_t size_sent = as_msg_send_fin(job->fd_h->sock, AS_OK);

			if (size_sent != 0) {
				job->net_io_ns += cf_getns() - before_ns;
				job->net_io_bytes += size_sent;
			}

			conn_query_job_release_fd(job, size_sent == 0);
		}
	}

	cf_mutex_destroy(&job->fd_lock);
}

static bool
conn_query_job_send_response(conn_query_job* job, uint8_t* buf, size_t size)
{
	as_query_job* _job = (as_query_job*)job;

	cf_mutex_lock(&job->fd_lock);

	if (job->fd_h == NULL) {
		cf_mutex_unlock(&job->fd_lock);
		return false;
	}

	uint64_t before_ns = cf_getns();
	size_t size_sent = send_blocking_response_chunk(job->fd_h, buf, size,
			job->fd_timeout, job->compress_response, NULL);

	if (size_sent != 0) {
		job->net_io_ns += cf_getns() - before_ns;
		job->net_io_bytes += size_sent;
	}

	cf_mutex_unlock(&job->fd_lock);

	if (size_sent == 0) {
		conn_query_job_release_fd(job, true);
		return false;
	}

	return true;
}

static void
conn_query_job_release_fd(conn_query_job* job, bool force_close)
{
	job->fd_h->last_used = cf_getns();
	as_end_of_transaction(job->fd_h, force_close);
	job->fd_h = NULL;
}

static size_t
send_blocking_response_chunk(as_file_handle* fd_h, uint8_t* buf, size_t size,
		int32_t timeout, bool compress, as_proto_comp_stat* comp_stat)
{
	cf_socket* sock = &fd_h->sock;
	as_proto* proto = (as_proto*)buf;

	proto->version = PROTO_VERSION;
	proto->type = PROTO_TYPE_AS_MSG;
	proto->sz = size - sizeof(as_proto);
	as_proto_swap(proto);

	const uint8_t* msgp = (const uint8_t*)buf;

	if (compress) {
		msgp = as_proto_compress(msgp, &size, comp_stat);
	}

	if (cf_socket_send_all(sock, msgp, size, MSG_NOSIGNAL, timeout) < 0) {
		cf_warning(AS_QUERY, "error sending to %s - fd %d sz %lu %s",
				fd_h->client, CSFD(sock), size, cf_strerror(errno));
		return 0;
	}

	return sizeof(as_proto) + size;
}
