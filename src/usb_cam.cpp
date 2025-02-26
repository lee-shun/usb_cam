#include "matplotlibcpp.h"
#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <numeric>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define DEVICE "/dev/video0"
#define WIDTH 1920
#define HEIGHT 1080
#define FPS 60
#define BUFFER_COUNT 4
namespace plt = matplotlibcpp;

// 实时性设置
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

struct FrameData {
public:
  uint64_t timestamp;
};

void analyze_and_plot_intervals(std::vector<FrameData> &frames,
                                std::string name = "1") {
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
  plt::save(name + "_frame_intervals.png");
  plt::close();
}
int main() {
  enable_realtime();

  // 打开设备
  int fd = open(DEVICE, O_RDWR);
  if (fd == -1) {
    perror("无法打开设备");
    exit(EXIT_FAILURE);
  }

  // 设置格式
  struct v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = WIDTH;
  fmt.fmt.pix.height = HEIGHT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;

  if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
    perror("设置格式失败");
    close(fd);
    exit(EXIT_FAILURE);
  }

  // 设置帧率
  struct v4l2_streamparm parm = {0};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = FPS;
  if (ioctl(fd, VIDIOC_S_PARM, &parm) == -1) {
    perror("设置帧率失败");
    close(fd);
    exit(EXIT_FAILURE);
  }

  // 请求缓冲区
  struct v4l2_requestbuffers req = {0};
  req.count = BUFFER_COUNT;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
    perror("请求缓冲区失败");
    close(fd);
    exit(EXIT_FAILURE);
  }

  // 映射缓冲区
  struct buffer {
    void *start;
    size_t length;
  } *buffers = (struct buffer *)calloc(BUFFER_COUNT, sizeof(*buffers));

  for (int i = 0; i < BUFFER_COUNT; ++i) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
      perror("查询缓冲区失败");
      close(fd);
      exit(EXIT_FAILURE);
    }

    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buf.m.offset);
    if (buffers[i].start == MAP_FAILED) {
      perror("内存映射失败");
      close(fd);
      exit(EXIT_FAILURE);
    }

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
      perror("入队缓冲区失败");
      close(fd);
      exit(EXIT_FAILURE);
    }
  }

  // 启动流
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
    perror("启动流失败");
    close(fd);
    exit(EXIT_FAILURE);
  }

  struct timespec next_cycle;
  clock_gettime(CLOCK_MONOTONIC, &next_cycle);
  const long loop_ns = 1000000000 / FPS;

  int i = 0, num_frames = 1000;
  std::vector<FrameData> frames, frames2;
  frames.reserve(num_frames);
  frames2.reserve(num_frames);

  while (i++ < num_frames) {
    // 获取帧数据
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
      perror("出队缓冲区失败");
      break;
    }

    // 获取精确时间戳
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    uint64_t host_us = ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    frames.push_back({host_us});

    uint64_t hw_us = 1e6 * buf.timestamp.tv_sec + buf.timestamp.tv_usec;
    frames2.push_back({hw_us});
    printf("the hw tp %ld us\n", hw_us);

    // 此处添加图像处理代码
    // process_image(buffers[buf.index].start);

    // 重新入队缓冲区
    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
      perror("重新入队失败");
      break;
    }

    // 计算下一周期
    next_cycle.tv_nsec += loop_ns;
    if (next_cycle.tv_nsec >= 1000000000) {
      next_cycle.tv_sec += 1;
      next_cycle.tv_nsec -= 1000000000;
    }

    // 精确等待
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_cycle, NULL);
  }

  // 清理资源
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd, VIDIOC_STREAMOFF, &type);
  for (int i = 0; i < BUFFER_COUNT; ++i)
    munmap(buffers[i].start, buffers[i].length);
  free(buffers);
  close(fd);

  analyze_and_plot_intervals(frames, "host_time");
  analyze_and_plot_intervals(frames2, "hw_time");

  return 0;
}
