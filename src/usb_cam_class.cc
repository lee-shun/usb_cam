#include "matplotlibcpp.h"
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// 配置参数
#define DEVICE_PATH "/dev/video0"
#define CAPTURE_WIDTH 1920
#define CAPTURE_HEIGHT 1080
#define PIXEL_FORMAT V4L2_PIX_FMT_YUYV
#define BUFFER_COUNT 8
#define FPS 60

using namespace std::chrono;
namespace plt = matplotlibcpp;

struct FrameBuffer {
  void *start;
  size_t length;
  uint32_t offset;
};

struct FrameData {
  uint64_t timestamp;
};

class V4L2Camera {
private:
  int fd_ = -1;
  FrameBuffer buffers_[BUFFER_COUNT];
  std::atomic<bool> running_{false};

  void xioctl(unsigned long request, void *arg, std::string error_msg = "") {
    int r;
    do {
      r = ioctl(fd_, request, arg);
    } while (-1 == r && EINTR == errno);
    if (r < 0) {
      std::cerr << error_msg << std::endl;
      close(fd_);
      throw std::runtime_error("ioctl failed: " + std::to_string(request));
    }
  }

  void enable_realtime() {
    struct sched_param param = {.sched_priority = 99};
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
      perror("sched_setscheduler failed");
      exit(EXIT_FAILURE);
    }

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
      perror("mlockall failed");
      exit(EXIT_FAILURE);
    }
  }

  cv::Mat yuyv_to_bgr(const void *yuyv_data) {
    cv::Mat yuyv(CAPTURE_HEIGHT, CAPTURE_WIDTH, CV_8UC2, (void *)yuyv_data);
    cv::Mat bgr;
    cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
    return bgr;
  }

  uint64_t getEpochTimeShift() {
    struct timeval epochtime;
    struct timespec vsTime;

    gettimeofday(&epochtime, NULL);
    clock_gettime(CLOCK_MONOTONIC, &vsTime);

    uint64_t uptime_us = vsTime.tv_sec * 1e6 + vsTime.tv_nsec / 1000;
    uint64_t epoch_us = epochtime.tv_sec * 1e6 + epochtime.tv_usec;
    return epoch_us - uptime_us;
  }

public:
  ~V4L2Camera() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

  void init() {
    // 1. 打开设备
    fd_ = open(DEVICE_PATH, O_RDWR);
    if (fd_ < 0) {
      throw std::runtime_error("Cannot open device: " +
                               std::string(DEVICE_PATH));
    }

    // 2. 设置格式
    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = CAPTURE_WIDTH;
    fmt.fmt.pix.height = CAPTURE_HEIGHT;
    fmt.fmt.pix.pixelformat = PIXEL_FORMAT;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    xioctl(VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT");

    // 3. 设置帧率
    v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = FPS;
    xioctl(VIDIOC_S_PARM, &parm, "VIDIOC_S_PARM");

    // 3. 请求缓冲区
    v4l2_requestbuffers req = {};
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(VIDIOC_REQBUFS, &req, "VIDIOC_REQBUFS");

    // 4. 内存映射
    for (uint32_t i = 0; i < req.count; ++i) {
      v4l2_buffer buf = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      xioctl(VIDIOC_QUERYBUF, &buf, "VIDIOC_QUERYBUF");

      buffers_[i].length = buf.length;
      buffers_[i].offset = buf.m.offset;
      buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd_, buf.m.offset);

      xioctl(VIDIOC_QBUF, &buf, "VIDIOC_QBUF");
    }

    // 5. 设置实时优先级
    enable_realtime();

    // 6. 开始采集
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
  }

  std::vector<FrameData> capture_frames(int num_frames) {
    std::vector<FrameData> frames;
    frames.reserve(num_frames);
    running_ = true;
    uint64_t toEpochOffset_us = getEpochTimeShift();

    while (frames.size() < num_frames && running_) {
      /* auto begin = high_resolution_clock::now();*/
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(fd_, &fds);

      timeval tv = {0, 10000}; // 10 ms timeout
      int r = select(fd_ + 1, &fds, NULL, NULL, &tv);

      if (r == -1 && errno == EINTR)
        continue;
      if (r == -1)
        throw std::runtime_error("select error");

      auto ts = high_resolution_clock::now();

      v4l2_buffer buf = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      xioctl(VIDIOC_DQBUF, &buf, "VIDIOC_DQBUF");
      uint64_t temp_us = 1e6 * buf.timestamp.tv_sec + buf.timestamp.tv_usec;
      printf("the hw tp %ld us\n", temp_us + toEpochOffset_us);

      // 转换格式+记录时间戳
      // cv::Mat bgr = yuyv_to_bgr(buffers_[buf.index].start);
      frames.push_back({temp_us + toEpochOffset_us});

      // cv::imshow("USB Camera", bgr);
      // cv::waitKey(1);

      xioctl(VIDIOC_QBUF, &buf, "VIDIOC_QBUF");

      /* auto end = high_resolution_clock::now();
      std::cout << "loop cost: "
                << duration_cast<milliseconds>(end - begin).count()
                << " ms, fps: "
                << 1000.0f / duration_cast<milliseconds>(end - begin).count()
                << std::endl; */
    }

    return frames;
  }

  void stop() {
    running_ = false;
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMOFF, &type);

    // 释放mmap内存
    for (auto &buf : buffers_) {
      if (buf.start)
        munmap(buf.start, buf.length);
    }
  }
};

// 验证函数
void analyze_and_plot_intervals(std::vector<FrameData> &frames) {
  if (frames.size() < 2) {
    std::cerr << "需要至少2帧数据" << std::endl;
    return;
  }

  // frames.erase(frames.begin());

  // 计算时间间隔（微秒）
  std::vector<double> intervals;
  for (size_t i = 1; i < frames.size(); ++i) {
    auto delta = frames[i].timestamp - frames[i - 1].timestamp;
    intervals.push_back(delta / 1000.0);
  }

  // 基础统计指标
  auto [min_it, max_it] =
      std::minmax_element(intervals.begin(), intervals.end());
  double min_val = *min_it;
  double max_val = *max_it;
  size_t min_idx = std::distance(intervals.begin(), min_it);
  size_t max_idx = std::distance(intervals.begin(), max_it);

  // 中位数计算
  std::vector<double> sorted = intervals;
  std::sort(sorted.begin(), sorted.end());
  double median = sorted[sorted.size() / 2];

  // 平均值与方差（无偏估计）
  double sum = std::accumulate(intervals.begin(), intervals.end(), 0.0);
  double mean = sum / intervals.size();
  double variance = 0.0;
  std::for_each(intervals.begin(), intervals.end(),
                [&](const double x) { variance += (x - mean) * (x - mean); });
  variance /= (intervals.size() - 1);

  // 平均帧率（帧/秒）
  auto total_duration = frames.back().timestamp - frames[0].timestamp;
  double avg_fps = (frames.size() - 1) / (total_duration / 1e6);

  // 输出统计结果
  std::cout << "=== 时间间隔分析 ===\n"
            << "最小值: " << min_val << " ms (帧间:" << min_idx << ")\n"
            << "最大值: " << max_val << " ms (帧间:" << max_idx << ")\n"
            << "中位数: " << median << " ms\n"
            << "平均值: " << mean << " ms\n"
            << "样本方差: " << variance << " ms²\n"
            << "平均帧率: " << avg_fps << " FPS\n";

  // 绘制二维图
  std::vector<int> x_axis(intervals.size());
  std::iota(x_axis.begin(), x_axis.end(), 1);
  plt::figure_size(1200, 600);
  plt::title("帧间隔时间分布");
  plt::plot(x_axis, intervals, "b-");
  plt::xlabel("间隔序号");
  plt::ylabel("时间间隔(ms)");
  plt::grid(true);
  plt::save("frame_intervals.png");
  plt::close();
}

int main() {
  try {
    V4L2Camera cam;
    cam.init();
    auto frames = cam.capture_frames(1000);
    cam.stop();
    analyze_and_plot_intervals(frames);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
