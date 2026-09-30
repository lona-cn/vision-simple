#include "HttpDispatch.h"

#include <hv/EventLoop.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace vision_simple {
struct HttpDispatch::Impl {
  struct Job {
    HttpContextPtr ctx;  // Accessed and finally reset only by the owning IO loop.
    hv::EventLoop* loop = nullptr;
    Work work;
    Reply reply;
    std::stop_source stop;
    Clock::time_point started;
    std::function<void()> previous_close;
    hclose_cb previous_io_close = nullptr;
    hread_cb previous_io_read = nullptr;
    std::function<void(hv::Buffer*)> previous_read;
    decltype(HttpMessage::http_cb) previous_http;
    bool body_complete = true;  // IO-loop-only, including streaming uploads.
    bool rejected_pipeline = false;
    bool disconnected = false;  // IO-loop-only.
    bool work_complete = false;  // IO-loop-only completion/close rendezvous.
    bool failed = false;
    size_t slot = 0;
  };
  struct Queue {
    size_t worker_count;
    size_t queue_capacity;
    size_t active = 0;
    size_t residents = 0;
    size_t head = 0;
    size_t pending = 0;
    std::vector<std::shared_ptr<Job>> slots;
    std::vector<size_t> fifo;
    std::vector<std::jthread> workers;
    Queue(size_t workers, size_t capacity)
        : worker_count(workers), queue_capacity(capacity),
          slots(workers + capacity), fifo(workers + capacity) {
      this->workers.reserve(workers);
    }
  };

  std::atomic<bool> accepting{true};
  std::mutex mutex;
  std::mutex stop_mutex;
  std::condition_variable changed;
  Queue data;
  Queue control;

  explicit Impl(Options options)
      : data(options.data_workers, options.data_queue_capacity),
        control(options.control_workers, options.control_queue_capacity) {}


  static void OnIoClose(hio_t* io) {
    // libhv skips writer.onclose for incomplete state-handler uploads. Hook
    // the actual socket close as well; the chained callback deletes its
    // HttpHandler, so do not access the Channel after invoking it.
    auto* channel = static_cast<hv::Channel*>(hio_context(io));
    if (channel && channel->id() == hio_id(io) && channel->onclose) {
      channel->status = hv::Channel::DISCONNECTED;
      channel->onclose();
    }
  }

  static void OnIoRead(hio_t* io, void* data, int size) {
    auto* channel = static_cast<hv::Channel*>(hio_context(io));
    if (channel && channel->id() == hio_id(io) && channel->onread) {
      hv::Buffer buffer(data, size);
      channel->onread(&buffer);
    }
  }

  static void RejectPipeline(const std::shared_ptr<Job>& job) {
    job->rejected_pipeline = true;
    job->stop.request_stop();
    // Do not delete HttpHandler while its parser is on the stack.
    job->ctx->writer->close(true);
  }
  void Start() {
    try {
      for (Queue* queue : {&data, &control})
        for (size_t i = 0; i < queue->worker_count; ++i)
          queue->workers.emplace_back([this, queue] { Run(*queue); });
    } catch (...) {
      BeginStop();
      data.workers.clear();
      control.workers.clear();
      throw;
    }
  }

  void Finish(Queue& queue, const std::shared_ptr<Job>& job) noexcept {
    // Even disconnected contexts remain resident until physical Work drain.
    // Restore before invoking Reply: End/close may synchronously invoke onclose.
    auto& ctx = job->ctx;
    job->work_complete = true;
    // close(true) may have marked the Channel closed before the real close
    // event deletes HttpHandler. Keep its hooked callback and context resident
    // until that event; otherwise its weak Job would expire and leak cleanup.
    if (!job->disconnected && !ctx->writer->isOpened()) return;
    ctx->writer->onclose = std::move(job->previous_close);
    ctx->writer->onread = std::move(job->previous_read);
    if (!job->disconnected && ctx->writer->isOpened()) {
      hio_setcb_close(ctx->writer->io(), job->previous_io_close);
      hio_setcb_read(ctx->writer->io(), job->previous_io_read);
      ctx->request->http_cb = std::move(job->previous_http);
    } else {
      // Its HttpHandler has died; never leave a callback to the deleted owner.
      ctx->request->http_cb = {};
      job->previous_http = {};
    }
    try {
      if (!job->disconnected && ctx->writer->isOpened()) {
        if (job->stop.stop_requested() || job->failed || !job->reply)
          ctx->writer->close();
        else
          job->reply(ctx);
      }
    } catch (...) {
      ctx->writer->close();
    }
    // Destroy all potentially context-owning captures on IO, before releasing
    // the resident slot. EventLoop may copy its closure; those copies then only
    // own an empty Job, never the last reference to a live Channel.
    job->reply = {};
    job->work = {};
    job->ctx.reset();
    {
      std::lock_guard lock(mutex);
      queue.slots[job->slot].reset();
      --queue.residents;
      // Stop may destroy Impl as soon as it observes zero; the notification
      // must be our last Impl access and remain inside this locked boundary.
      changed.notify_all();
    }
  }

  void Run(Queue& queue) noexcept {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] {
          return queue.pending != 0 || !accepting.load(std::memory_order_relaxed);
        });
        if (!queue.pending) return;
        const size_t slot = queue.fifo[queue.head];
        queue.head = (queue.head + 1) % queue.fifo.size();
        --queue.pending;
        ++queue.active;
        job = queue.slots[slot];
      }
      // A disconnected/shutdown queued request must not load a model merely
      // to discover cancellation. Active work receives its actual stop token.
      if (!accepting.load(std::memory_order_acquire)) job->stop.request_stop();
      if (!job->stop.stop_requested()) {
        try {
          job->reply = job->work(job->stop.get_token(), job->started);
        } catch (...) {
          job->failed = true;
        }
      }
      {
        std::lock_guard lock(mutex);
        --queue.active;
      }
      // Posting allocation failure cannot safely release a live Channel on
      // this worker or pretend successful drain. Fail fast rather than violate
      // thread ownership. EventLoop's API itself has no recoverable post result.
      try {
        job->loop->queueInLoop([this, &queue, job = std::move(job)] {
          Finish(queue, job);
        });
      } catch (...) {
        std::terminate();
      }
    }
  }

  bool Submit(const HttpContextPtr& ctx, Lane lane, Work work,
              bool body_complete) noexcept {
    if (!accepting.load(std::memory_order_acquire) || !ctx || !ctx->request ||
        !ctx->writer || !ctx->writer->isOpened() || !work)
      return false;
    auto* loop = hv::tlsEventLoop();
    if (!loop || !loop->isInLoopThread() ||
        hevent_loop(ctx->writer->io()) != loop->loop() ||
        hio_context(ctx->writer->io()) != static_cast<hv::Channel*>(ctx->writer.get()))
      return false;
    Queue& queue = lane == Lane::kData ? data : control;
    try {
      auto job = std::make_shared<Job>();
      job->loop = loop;
      job->work = std::move(work);
      job->started = Clock::now();
      job->body_complete = body_complete;
      std::weak_ptr<Job> weak = job;
      std::function<void()> closed = [this, &queue, weak] {
        if (auto live = weak.lock(); live && !live->disconnected) {
          live->disconnected = true;
          live->stop.request_stop();
          try {
            if (live->previous_close) live->previous_close();
          } catch (...) {
            // Preserve libhv socket/handler cleanup even if an old hook throws.
          }
          if (live->previous_io_close)
            live->previous_io_close(live->ctx->writer->io());
          if (live->work_complete) Finish(queue, live);
        }
      };
      std::function<void(hv::Buffer*)> read = [weak](hv::Buffer* buffer) {
        if (auto live = weak.lock(); live && !live->disconnected) {
          if (live->body_complete || live->rejected_pipeline) {
            RejectPipeline(live);
          } else if (live->previous_io_read) {
            live->previous_io_read(live->ctx->writer->io(),
                                   buffer->data(), static_cast<int>(buffer->size()));
          }
        }
      };
      decltype(job->previous_http) http =
          [weak](HttpMessage* message, http_parser_state phase,
                 const char* bytes, size_t size) {
            if (auto live = weak.lock(); live && !live->disconnected &&
                !live->rejected_pipeline) {
              if (phase == HP_MESSAGE_BEGIN && live->body_complete) {
                RejectPipeline(live);
                return;
              }
              if (phase == HP_MESSAGE_COMPLETE) live->body_complete = true;
              if (live->previous_http)
                live->previous_http(message, phase, bytes, size);
            }
          };
      {
        std::lock_guard lock(mutex);
        if (!accepting.load(std::memory_order_relaxed) ||
            queue.residents == queue.slots.size() ||
            queue.pending >= queue.worker_count - queue.active + queue.queue_capacity)
          return false;
        size_t slot = 0;
        while (queue.slots[slot]) ++slot;
        job->slot = slot;
        job->ctx = ctx;
        job->previous_close = std::move(ctx->writer->onclose);
        job->previous_io_close = hio_getcb_close(ctx->writer->io());
        job->previous_io_read = hio_getcb_read(ctx->writer->io());
        job->previous_read = std::move(ctx->writer->onread);
        // Pinned libhv's http_cb captures only HttpHandler*. It accesses that
        // pointer before delegating to a member, and never accesses its target
        // again after the member returns. Moving it during Submit inside that
        // delegate is safe; unknown user callbacks are not accepted here.
        job->previous_http = std::move(ctx->request->http_cb);
        ctx->request->http_cb = std::move(http);
        ctx->writer->onread = std::move(read);
        hio_setcb_read(ctx->writer->io(), OnIoRead);
        ctx->writer->onclose = std::move(closed);
        hio_setcb_close(ctx->writer->io(), OnIoClose);
        queue.slots[slot] = std::move(job);
        queue.fifo[(queue.head + queue.pending) % queue.fifo.size()] = slot;
        ++queue.pending;
        ++queue.residents;
      }
      changed.notify_all();
      return true;
    } catch (...) {
      // All throwing construction precedes touching callbacks/admission.
      return false;
    }
  }

  void BeginStop() noexcept {
    {
      std::lock_guard lock(mutex);
      accepting.store(false, std::memory_order_release);
    }
    // request_stop executes user stop_callbacks synchronously; never execute
    // them under the admission lock. Fixed slots avoid a shutdown allocation.
    for (Queue* queue : {&data, &control}) {
      for (size_t slot = 0; slot < queue->slots.size(); ++slot) {
        std::shared_ptr<Job> job;
        {
          std::lock_guard lock(mutex);
          job = queue->slots[slot];
        }
        if (job) job->stop.request_stop();
      }
    }
    changed.notify_all();
  }

  void Stop() noexcept {
    std::lock_guard stop_lock(stop_mutex);
    BeginStop();
    data.workers.clear();
    control.workers.clear();
    std::unique_lock lock(mutex);
    changed.wait(lock, [&] { return data.residents == 0 && control.residents == 0; });
  }
};

HttpDispatch::HttpDispatch(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
HttpDispatch::~HttpDispatch() { Stop(); }
VSResult<std::unique_ptr<HttpDispatch>> HttpDispatch::Create(Options options) noexcept {
  if (!options.data_workers || options.data_workers > 32 ||
      !options.control_workers || options.control_workers > 32 ||
      !options.data_queue_capacity || options.data_queue_capacity > 128 ||
      !options.control_queue_capacity || options.control_queue_capacity > 128)
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "Invalid HTTP dispatch options");
  try {
    auto dispatch = std::unique_ptr<HttpDispatch>(
        new HttpDispatch(std::make_unique<Impl>(options)));
    dispatch->impl_->Start();
    return dispatch;
  } catch (...) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                      "Unable to start HTTP dispatch workers");
  }
}
bool HttpDispatch::Submit(const HttpContextPtr& ctx, Lane lane, Work work,
                          bool body_complete) noexcept {
  return impl_->Submit(ctx, lane, std::move(work), body_complete);
}
bool HttpDispatch::Accepting() const noexcept {
  return impl_->accepting.load(std::memory_order_acquire);
}
void HttpDispatch::BeginStop() noexcept { impl_->BeginStop(); }
void HttpDispatch::Stop() noexcept { impl_->Stop(); }
}  // namespace vision_simple
