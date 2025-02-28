__kernel void yuyv_to_rgb(
    __global const uchar* yuyv_buffer,
    __global uchar* rgb_buffer,
    int width,
    int height)
{
    // 获取当前工作项的全局坐标
    int x = get_global_id(0);
    int y = get_global_id(1);

    // 边界检查，确保不越界
    if (x >= width / 2 || y >= height)
        return;

    // 计算输入缓冲区索引并读取YUYV数据块
    int block_index = y * (width / 2) + x;
    uchar4 yuyv = vload4(block_index, yuyv_buffer);

    // 提取YUV分量
    uchar Y0 = yuyv.s0;
    uchar U  = yuyv.s1;
    uchar Y1 = yuyv.s2;
    uchar V  = yuyv.s3;

    // 将U/V转换为有符号浮点数 (-128 ~ 127)
    float u = (float)U - 128.0f;
    float v = (float)V - 128.0f;

    // Y分量转为浮点数 (全范围 0-255)
    float y0 = (float)Y0;
    float y1 = (float)Y1;

    // 计算第一个像素的RGB值
    float r0 = y0 + 1.403f * v;
    float g0 = y0 - 0.344f * u - 0.714f * v;
    float b0 = y0 + 1.773f * u;

    // 计算第二个像素的RGB值
    float r1 = y1 + 1.403f * v;
    float g1 = y1 - 0.344f * u - 0.714f * v;
    float b1 = y1 + 1.773f * u;

    // 将浮点数值限制在0-255并转换为uchar
    uchar3 rgb_pixel0 = (uchar3)(
        clamp(r0, 0.0f, 255.0f),
        clamp(g0, 0.0f, 255.0f),
        clamp(b0, 0.0f, 255.0f)
    );

    uchar3 rgb_pixel1 = (uchar3)(
        clamp(r1, 0.0f, 255.0f),
        clamp(g1, 0.0f, 255.0f),
        clamp(b1, 0.0f, 255.0f)
    );

    // 计算输出缓冲区索引
    int out_index = (y * width + 2 * x) * 3;

    // 使用vstore3进行高效存储
    vstore3(rgb_pixel0, 0, rgb_buffer + out_index);
    vstore3(rgb_pixel1, 0, rgb_buffer + out_index + 3);
}
