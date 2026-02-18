#version 450
layout (local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer OutputImage {
    vec4 pixels[];
};

layout(push_constant) uniform PushConstants {
    int width;
    int height;
} push;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= push.width || y >= push.height) return;

    uint idx = y * push.width + x;

    // For now just a gradient test
    float fx = float(x) / float(push.width);
    float fy = float(y) / float(push.height);
    pixels[idx] = vec4(fx, fy, 0.5, 1.0);
}
