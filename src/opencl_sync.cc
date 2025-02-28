#include <CL/cl2.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class YUYVConverter {
public:
  YUYVConverter(int width, int height) : width_(width), height_(height) {
    setupOpenCL();
  }

  void convert(const cv::Mat &yuyv, cv::Mat &rgb) {
    // 上传数据到设备
    queue_.enqueueWriteBuffer(yuyvBuffer_, CL_TRUE, 0,
                              yuyv.total() * yuyv.elemSize(), yuyv.data);

    // 设置内核参数
    kernel_.setArg(0, yuyvBuffer_);
    kernel_.setArg(1, rgbBuffer_);
    kernel_.setArg(2, width_);
    kernel_.setArg(3, height_);

    // 执行内核
    cl::NDRange global(width_ / 2, height_);
    queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, global, cl::NullRange);
    queue_.finish();

    // 读取结果
    queue_.enqueueReadBuffer(rgbBuffer_, CL_TRUE, 0,
                             rgb.total() * rgb.elemSize(), rgb.data);
  }

private:
  void setupOpenCL() {
    // 获取OpenCL平台和设备
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    device_ = devices[0];

    // 创建上下文和命令队列
    context_ = cl::Context(device_);
    queue_ = cl::CommandQueue(context_, device_);

    // 创建缓冲区
    yuyvBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, width_ * height_ * 2);
    rgbBuffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, width_ * height_ * 3);

    // 编译内核程序
    const char *kernelCode = R"(
            __kernel void yuyv_to_rgb(__global const uchar* yuyv_buffer, 
                                    __global uchar* rgb_buffer,
                                    int width,
                                    int height) 
            {
                int x = get_global_id(0);
                int y = get_global_id(1);

                if (x >= width / 2 || y >= height)
                    return;

                int block_index = y * (width / 2) + x;
                uchar4 yuyv = vload4(block_index, yuyv_buffer);

                uchar Y0 = yuyv.s0;
                uchar U  = yuyv.s1;
                uchar Y1 = yuyv.s2;
                uchar V  = yuyv.s3;

                float u = (float)U - 128.0f;
                float v = (float)V - 128.0f;

                float y0 = (float)Y0;
                float y1 = (float)Y1;

                float r0 = y0 + 1.403f * v;
                float g0 = y0 - 0.344f * u - 0.714f * v;
                float b0 = y0 + 1.773f * u;

                float r1 = y1 + 1.403f * v;
                float g1 = y1 - 0.344f * u - 0.714f * v;
                float b1 = y1 + 1.773f * u;

                uchar3 rgb0 = (uchar3)(
                    clamp(r0, 0.0f, 255.0f),
                    clamp(g0, 0.0f, 255.0f),
                    clamp(b0, 0.0f, 255.0f)
                );

                uchar3 rgb1 = (uchar3)(
                    clamp(r1, 0.0f, 255.0f),
                    clamp(g1, 0.0f, 255.0f),
                    clamp(b1, 0.0f, 255.0f)
                );

                int out_index = (y * width + 2 * x) * 3;
                vstore3(rgb0, 0, rgb_buffer + out_index);
                vstore3(rgb1, 0, rgb_buffer + out_index + 3);
            }
        )";

    cl::Program::Sources sources;
    sources.push_back({kernelCode, strlen(kernelCode)});

    program_ = cl::Program(context_, sources);
    std::cerr << "构建日志:\n"
              << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_)
              << std::endl;
    if (program_.build({device_}) != CL_SUCCESS) {
      throw std::runtime_error("OpenCL程序构建失败");
    }

    kernel_ = cl::Kernel(program_, "yuyv_to_rgb");
  }

  int width_, height_;
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;
  cl::Kernel kernel_;
  cl::Buffer yuyvBuffer_, rgbBuffer_;
};

int main() {
  const int width = 1920;
  const int height = 1080;

  // 打开视频设备（YUYV格式）
  cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "无法打开视频设备" << std::endl;
    return -1;
  }

  // 设置视频格式为YUYV
  cap.set(cv::CAP_PROP_CONVERT_RGB, false);
  cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

  // 初始化转换器
  YUYVConverter converter(width, height);

  cv::Mat yuyvFrame(height, width, CV_8UC2);
  cv::Mat rgbFrame(height, width, CV_8UC3);

  while (true) {
    cap >> yuyvFrame;
    if (yuyvFrame.empty())
      break;

    // OpenCL转换
    converter.convert(yuyvFrame, rgbFrame);

    // 转换颜色空间（BGR用于显示）
    cv::Mat displayFrame;

    // 显示结果
    cv::imshow("Preview", rgbFrame);
    if (cv::waitKey(1) == 27)
      break; // ESC退出
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
