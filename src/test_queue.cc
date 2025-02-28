#include <CL/cl2.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <list>
#include <mutex>
#include <queue>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

/**获取编译program出错时，编译器的出错信息*/
int getProgramBuildInfo(cl_program program, cl_device_id device) {
  size_t log_size;
  char *program_log;
  /* Find size of log and print to std output */
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                        &log_size);
  program_log = (char *)malloc(log_size + 1);
  program_log[log_size] = '\0';
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1,
                        program_log, NULL);
  printf("%s\n", program_log);
  free(program_log);
  return 0;
}

// 帧元数据结构体
struct VideoFrame {
  std::vector<uint8_t> data;
  uint64_t capture_ts; // 采集时间（微秒）
  uint64_t process_ts; // OpenCL处理完成时间
  uint64_t display_ts; // 显示时间
};

// 线程安全队列模板
template <typename T> class SafeQueue {
public:
  SafeQueue(size_t max_size = 10) : max_size_(max_size) {}

  // 通用引用版本的 push
  template <typename U> bool push(U &&item) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_push_.wait(lock, [this]() { return queue_.size() < max_size_; });

    queue_.push(std::forward<U>(item)); // 完美转发
    cv_pop_.notify_one();
    return true;
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!cv_pop_.wait_for(lock, std::chrono::milliseconds(100),
                          [this]() { return !queue_.empty(); })) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop();
    cv_push_.notify_one();
    return true;
  }

private:
  std::queue<T> queue_;
  mutable std::mutex mtx_;
  std::condition_variable cv_push_, cv_pop_;
  size_t max_size_;
};

class VideoPipeline {
public:
  VideoPipeline(const char *cam, int width, int height)
      : width_(width), height_(height), running_(true) {

    // 初始化V4L2
    init_v4l2(cam);

    // 初始化OpenCL
    cl::Device device = cl::Device::getDefault();
    context_ = cl::Context(device);
    queue_ = cl::CommandQueue(context_, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                            CL_QUEUE_PROFILING_ENABLE);

    // 创建三重缓冲
    for (int i = 0; i < 3; ++i) {
      cl_buffers_[i] =
          cl::Buffer(context_, CL_MEM_READ_WRITE, width_ * height_ * 3);
      free_buffers_.push(i);
    }

    // 编译内核
    std::ifstream kernel_file("yuv_rgb.cl");
    std::string src(std::istreambuf_iterator<char>(kernel_file),
                    (std::istreambuf_iterator<char>()));
    program_ = cl::Program(context_, src);
    std::cout << "before compile" << std::endl;
    program_.build();
    std::cout << "after compile" << std::endl;
    // program_.getBuildInfo(device);
    kernel_ = cl::Kernel(program_, "yuv2rgb");

    // 启动线程
    capture_thread_ = std::thread(&VideoPipeline::capture_loop, this);
    process_thread_ = std::thread(&VideoPipeline::process_loop, this);
  }

  ~VideoPipeline() {
    running_ = false;
    capture_thread_.join();
    process_thread_.join();
  }

  bool get_frame(VideoFrame &frame) { return output_queue_.pop(frame); }

private:
  struct CallbackData {
    VideoPipeline *pipeline;
    int cl_buf_idx;
    VideoFrame frame;
  };

  std::list<std::shared_ptr<CallbackData>> active_cbdatas_;
  std::mutex callback_mutex_;

  static void CL_CALLBACK read_complete_callback(cl_event event, cl_int status,
                                                 void *user_data) {
    auto *data = static_cast<CallbackData *>(user_data);
    if (status != CL_COMPLETE) {
      std::cerr << "OpenCL not completed!" << std::endl;
      return;
    }
    data->pipeline->handle_gpu_complete(data->cl_buf_idx,
                                        std::move(data->frame));

    std::lock_guard<std::mutex> lock(data->pipeline->callback_mutex_);
    data->pipeline->active_cbdatas_.remove_if(
        [data](const auto &ptr) { return ptr.get() == data; });
    std::cout << "in callback" << std::endl;
  }

  void handle_gpu_complete(int cl_buf_idx, VideoFrame &&frame) {
    frame.process_ts = now_us();
    output_queue_.push(std::move(frame));
    free_cl_buffers_.push(cl_buf_idx);
  }

  void init_v4l2(const char *device) {
    fd_ = open(device, O_RDWR);

    // 设置格式
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
      perror("设置格式失败");
      close(fd_);
      exit(EXIT_FAILURE);
    }

    // 设置帧率
    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 60;
    if (ioctl(fd_, VIDIOC_S_PARM, &parm) == -1) {
      perror("设置帧率失败");
      close(fd_);
      exit(EXIT_FAILURE);
    }

    // 请求缓冲
    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(VIDIOC_REQBUFS, &req);

    // 内存映射
    buffers_.resize(req.count);
    for (int i = 0; i < req.count; ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      xioctl(VIDIOC_QUERYBUF, &buf);

      buffers_[i].length = buf.length;
      buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd_, buf.m.offset);
    }

    // 启动采集
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMON, &type);
  }

  void capture_loop() {
    while (running_) {
      // 获取空闲缓冲区
    std::cout << "in capture loop" << std::endl;
      int buf_idx;
      if (!free_buffers_.pop(buf_idx)) {
        std::cout << "no free buffers" << std::endl;
        continue;
      }

      // 采集帧
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = buf_idx;
      xioctl(VIDIOC_DQBUF, &buf);

      // 记录时间戳
      VideoFrame frame;
      frame.capture_ts =
          buf.timestamp.tv_sec * 1000000ULL + buf.timestamp.tv_usec;
      frame.data.assign(static_cast<uint8_t *>(buffers_[buf_idx].start),
                        static_cast<uint8_t *>(buffers_[buf_idx].start) +
                            buf.bytesused);

      // 提交到处理队列
      input_queue_.push(std::move(frame));
      std::cout << "input queue size " << input_queue_.size() << std::endl;

      // 重新入队缓冲
      xioctl(VIDIOC_QBUF, &buf);
    }
  }

  void process_loop() {
    std::vector<cl::Event> events(3);
    int active_buf = 0;

    while (running_) {
      VideoFrame frame;
      if (!input_queue_.pop(frame))
        continue;

      // 获取OpenCL缓冲
      int cl_buf_idx;
      if (!free_cl_buffers_.pop(cl_buf_idx))
        continue;

      // 上传数据
      cl::Event upload_event;
      queue_.enqueueWriteBuffer(cl_buffers_[cl_buf_idx], CL_FALSE, 0,
                                frame.data.size(), frame.data.data(), nullptr,
                                &upload_event);

      // 执行内核
      cl::Event kernel_event;
      kernel_.setArg(0, cl_buffers_[cl_buf_idx]);
      kernel_.setArg(1, cl_buffers_[cl_buf_idx]);
      kernel_.setArg(2, width_);
      kernel_.setArg(3, height_);
      std::vector<cl::Event> up_events = {upload_event};
      queue_.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                  cl::NDRange(width_, height_), cl::NullRange,
                                  &events, &kernel_event);

      // 下载结果
      cl::Event read_event;
      std::vector<cl::Event> kl_events = {kernel_event};
      queue_.enqueueReadBuffer(cl_buffers_[cl_buf_idx], CL_FALSE, 0,
                               frame.data.size(), frame.data.data(), &kl_events,
                               &read_event);

      auto cb_data = std::make_shared<CallbackData>();
      cb_data->pipeline = this;
      cb_data->cl_buf_idx = cl_buf_idx;
      cb_data->frame = std::move(frame);
      // 注册完成回调
      read_event.setCallback(
          CL_COMPLETE, &VideoPipeline::read_complete_callback, cb_data.get());

      {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        active_cbdatas_.push_back(cb_data);
      }

      active_buf = (active_buf + 1) % 3;
    }
  }

  template <typename T> void xioctl(unsigned long request, T *arg) {
    if (ioctl(fd_, request, arg) < 0) {
      throw std::runtime_error("V4L2 ioctl failed");
    }
  }

  static uint64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
  }

  // 成员变量
  int width_, height_;
  std::atomic<bool> running_;

  // V4L2相关
  int fd_;
  struct Buffer {
    void *start;
    size_t length;
  };
  std::vector<Buffer> buffers_;

  // OpenCL相关
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;
  cl::Kernel kernel_;
  std::array<cl::Buffer, 3> cl_buffers_;

  // 队列系统
  SafeQueue<int> free_buffers_{4};     // 空闲V4L2缓冲
  SafeQueue<int> free_cl_buffers_{3};  // 空闲OpenCL缓冲
  SafeQueue<VideoFrame> input_queue_;  // 原始帧输入队列
  SafeQueue<VideoFrame> output_queue_; // 处理完成队列

  // 线程
  std::thread capture_thread_, process_thread_;
  mutable std::mutex stats_mutex_;
};

int main() {
  try {
    VideoPipeline pipeline("/dev/video0", 1920, 1080);

    // 统计信息
    uint64_t total_latency = 0;
    uint64_t frame_count = 0;

    while (true) {
      VideoFrame frame;

      if (pipeline.get_frame(frame)) {
        // 计算延迟
        uint64_t e2e_latency = frame.display_ts - frame.capture_ts;
        total_latency += e2e_latency;
        frame_count++;

        if (frame_count % 30 == 0) {
          std::cout << "平均延迟: " << total_latency / frame_count << "μs\n";
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
