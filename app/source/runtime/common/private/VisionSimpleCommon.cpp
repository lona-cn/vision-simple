#include "VisionSimpleCommon.h"
#include "LogFacade.h"

// LogContext 的 thread_local traceId 存储
thread_local std::string vision_simple::LogContext::current_trace_id_;

// LogFacade 的全局 sink 指针
vision_simple::LogSink* vision_simple::LogFacade::sink_ = nullptr;

