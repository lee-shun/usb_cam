#include "matplotlibcpp.h"
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>
namespace plt = matplotlibcpp;

// 平台特定的微秒级精确休眠实现
void precise_usleep(long us) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  long target_ns = us * 1000;
  while (true) {
    clock_gettime(CLOCK_MONOTONIC, &end);
    long elapsed = (end.tv_sec - start.tv_sec) * 1000000000L +
                   (end.tv_nsec - start.tv_nsec);
    if (elapsed >= target_ns)
      break;
    if (target_ns - elapsed > 10000) {
      struct timespec req = {0, target_ns - elapsed};
      nanosleep(&req, nullptr); // 大于10微秒使用系统休眠
    } else {
      __asm__ __volatile__("pause" ::: "memory"); // 短时忙等
    }
  }
}

void plot_results(const std::vector<long> &latencies, double avg,
                  long max_lat) {
  // 直方图参数配置
  const int bins = 50;
  const std::string color = "#1f77b4";

  plt::figure_size(1000, 600);       // 设置图像尺寸
  plt::hist(latencies, bins, color); // 绘制直方图

  // 添加统计标注
  plt::text(0.7, 0.8,
            "Average: " + std::to_string(avg) +
                "μs\nMax: " + std::to_string(max_lat) + "μs");

  // 图例配置
  plt::title("Latency Distribution (100μs Target)");
  plt::xlabel("Latency (μs)");
  plt::ylabel("Frequency");
  plt::grid(true);

  // 保存图像文件
  plt::save("./latency_distribution.png", 300); // 300dpi分辨率
  plt::close();
}

void latency_test() {
  using namespace std::chrono;
  constexpr int ITERATIONS = 1000; // C++14数字分隔符增强可读性
  std::vector<long> latencies;
  latencies.reserve(ITERATIONS);

  // 预热操作（减少首次调用的调度影响）
  for (int i = 0; i < 1000; ++i) {
    precise_usleep(100);
  }

  // 主测试循环
  for (int i = 0; i < ITERATIONS; ++i) {
    auto t1 = high_resolution_clock::now();
    precise_usleep(100);
    auto t2 = high_resolution_clock::now();
    latencies.push_back(duration_cast<microseconds>(t2 - t1).count());
  }

  // 统计计算
  const auto max_lat = *std::max_element(latencies.begin(), latencies.end());
  const double avg =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) / ITERATIONS;

  // 输出结果＆生成图表
  printf("Average: %.2fμs | Maximum: %ldμs\n", avg, max_lat);
  plot_results(latencies, avg, max_lat);
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

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage:sudo ./%s [0|1]: 0 for none enabled lock memory and "
           "scheduler fifo, 1 for enabled\n ",
           argv[0]);
    return 1;
  }

  if (argv[1][0] == '1') {
    printf("Lock memory and scheduler fifo enabled\n");
    enable_realtime();
  }
  latency_test();
  return 0;
}
