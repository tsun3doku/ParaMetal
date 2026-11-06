#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(push_constant) uniform PushConstants {
    vec4 worldPos;      // xyz = worldPosition, w = scale (default 1.0)
    vec4 colorTint;     // rgba tint
    vec2 viewportSize;  // width, height in pixels
    vec2 markerSize;    // width, height in pixels (e.g. 34.0, 34.0)
    uint isSelected;    // 0 = unselected, 1 = selected
    uint pad0;
    uint pad1;
    uint pad2;
} pc;

layout(location = 0) out vec2 fragTexCoord;

// Quad vertices generated directly from gl_VertexIndex (6 vertices = 2 triangles)
// The texture is 128x128 with the pin scaled to 96px height and tip at y=112 (UV 0.875).
// Needle tip anchor is at localOffset (0.0, 0.0) -> pin tip precisely touches worldPos
// X in [-0.5, 0.5], Y in [-0.125, 0.875] (total height = 1.0, Y=0.0 is the tip)
const vec2 quadOffsets[6] = vec2[](
    vec2(-0.5,  0.875), // top-left
    vec2(-0.5, -0.125), // bottom-left (below tip margin)
    vec2( 0.5, -0.125), // bottom-right (below tip margin)
    vec2( 0.5, -0.125), // bottom-right
    vec2( 0.5,  0.875), // top-right
    vec2(-0.5,  0.875)  // top-left
);

const vec2 quadUVs[6] = vec2[](
    vec2(0.0, 0.0), // top-left
    vec2(0.0, 1.0), // bottom-left
    vec2(1.0, 1.0), // bottom-right
    vec2(1.0, 1.0), // bottom-right
    vec2(1.0, 0.0), // top-right
    vec2(0.0, 0.0)  // top-left
);

void main() {
    vec2 localOffset = quadOffsets[gl_VertexIndex];
    fragTexCoord = quadUVs[gl_VertexIndex];

    // Transform 3D surface point to clip space
    vec4 clipPos = ubo.proj * ubo.view * vec4(pc.worldPos.xyz, 1.0);

    // Marker dimensions in pixels
    vec2 sizePixels = pc.markerSize * pc.worldPos.w;

    // Screen-space NDC offset
    // In Vulkan, NDC spans [-1, 1], with negative Y pointing upwards on screen
    vec2 ndcOffset = vec2(
        (localOffset.x * sizePixels.x / pc.viewportSize.x) * 2.0,
        -(localOffset.y * sizePixels.y / pc.viewportSize.y) * 2.0
    );

    // Apply offset in clip space (multiplied by clipPos.w so it remains invariant to perspective distance)
    clipPos.xy += ndcOffset * clipPos.w;

    // Very minimally pull depth closer to camera to prevent z-fighting / clipping with surface geometry
    clipPos.z = max(0.0, clipPos.z - 0.0005 * clipPos.w);

    gl_Position = clipPos;
}

