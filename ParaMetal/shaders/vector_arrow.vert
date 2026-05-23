#version 450

layout(location = 0) in vec3 localPos;

struct SurfacePoint {
    vec3 position;
    float temperature;
    vec3 normal;
    float area;
    vec4 color;
};

layout(binding = 0) readonly buffer SurfacePoints {
    SurfacePoint surfacePoints[];
};

layout(binding = 1) readonly buffer GradientColors {
    vec4 gradientColors[];
};

layout(binding = 2) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    float scale;
    float normalOffset;
    float minLength;
    float baseLength;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    // Read gradient vector from gradient buffer
    vec4 gradientData = gradientColors[gl_InstanceIndex];
    vec3 vectorModel = gradientData.xyz;
    float magnitude = length(vectorModel);

    float scaledLength = magnitude * pc.scale;
    if (scaledLength < pc.minLength || pc.scale <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
        outColor = vec4(0.0);
        return;
    }

    // Read position and normal from surface buffer
    SurfacePoint surfacePoint = surfacePoints[gl_InstanceIndex];

    mat3 normalMatrix = mat3(pc.modelMatrix);
    vec3 direction = normalize(normalMatrix * vectorModel);
    vec3 normal = normalize(normalMatrix * surfacePoint.normal);
    if (length(normal) <= 1e-5) {
        normal = vec3(0.0, 1.0, 0.0);
    }

    vec3 side = cross(normal, direction);
    if (length(side) <= 1e-5) {
        vec3 fallback = abs(direction.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        side = cross(fallback, direction);
    }
    side = normalize(side);
    vec3 up = normalize(cross(direction, side));

    float maxLength = pc.baseLength * pc.scale;
    float arrowLength = min(scaledLength, maxLength);
    vec3 origin = (pc.modelMatrix * vec4(surfacePoint.position, 1.0)).xyz + normal * pc.normalOffset;
    vec3 worldPos =
        origin +
        direction * localPos.x * arrowLength +
        side * localPos.y * arrowLength +
        up * localPos.z * arrowLength;

    gl_Position = ubo.proj * ubo.view * vec4(worldPos, 1.0);

    outColor = vec4(0.0, 0.304, 0.918, 0.95);
}
