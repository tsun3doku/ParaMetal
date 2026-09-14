#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(vec3(0.5, 0.8, 0.6));
    float diff = abs(dot(N, L)) * 0.5 + 0.5;
    vec3 baseColor = vec3(0.0, 0.85, 0.95);
    outColor = vec4(baseColor * diff, 0.75);
}
