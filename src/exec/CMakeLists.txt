target_sources(
  sirius_objects
  PRIVATE arrow_result_stream.cpp batch_stream.cpp stream_session.cpp
          stream_bind_catalog.cpp stream_plan_bindings.cpp
          streaming_fragment.cpp)

target_sources(
  sirius_objects
  PRIVATE starrocks_stream_session.cpp
          starrocks_streaming_fragment.cpp
          starrocks_substrait_compat.cpp
          exchange_inbox.cpp
          exchange_staging.cpp
          exchange_ring.cpp
          exchange_ring_ends.cpp
          exchange_ring_wire.cpp)
