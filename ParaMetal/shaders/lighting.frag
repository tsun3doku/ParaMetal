#version 450
#extension GL_KHR_vulkan_glsl:enable

layout(set = 0, binding = 4) uniform UniformBufferObject {
    mat4 model; 
    mat4 view;  
    mat4 proj;
    vec3 color;
} ubo;

layout(set = 0, binding = 5) uniform LightUniformBufferObject {
    vec4 iblParams; // x=intensity, y=diffuseScale, z=specularScale, w=maxReflectionLod
    vec3 cameraPos;
} lightUbo;

// Input attachments for GBuffer data
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputAlbedo;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput inputNormal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput inputPosition;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput inputMaterial;

// IBL environment maps
layout(set = 0, binding = 6) uniform samplerCube irradianceMap;    // Convolved diffuse irradiance
layout(set = 0, binding = 7) uniform samplerCube prefilteredMap;   // Specular env map, mipped by roughness
layout(set = 0, binding = 8) uniform sampler2D   brdfLut;          // Split sum BRDF integration LUT

layout(location = 0) in vec2 inUV; 

// Output color
layout(location = 0) out vec4 fragColor;

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Main

void main() {
    // Read GBuffer data
    vec4 albedoSample   = subpassLoad(inputAlbedo);
    float coverage = clamp(albedoSample.a, 0.0, 1.0);
    if (coverage <= 0.0) {
        // No geometry coverage at this sample
        discard;
    }

    vec4 normalSample   = subpassLoad(inputNormal);
    vec4 positionSample = subpassLoad(inputPosition);
    vec4 materialSample = subpassLoad(inputMaterial);

    float invCoverage = 1.0 / max(coverage, 1e-5);
    vec3 albedo    = albedoSample.rgb * invCoverage;
    vec3 normal    = normalize(normalSample.rgb);
    vec3 position  = positionSample.rgb * invCoverage;
    vec4 material  = materialSample * invCoverage;
    float roughness = clamp(material.r, 0.04, 1.0);
    float metalness = clamp(material.g, 0.0,  1.0);
    float lightingMix = clamp(material.b, 0.0, 1.0);

    vec3 viewDir = normalize(lightUbo.cameraPos - position);
    float NdotV  = max(dot(normal, viewDir), 0.0);

    // Derive F0 from metalness
    vec3 F0 = mix(vec3(0.04), albedo, metalness);

    // IBL lighting
    float iblIntensity = max(lightUbo.iblParams.x, 0.0);
    float iblDiffuseScale = max(lightUbo.iblParams.y, 0.0);
    float iblSpecularScale = max(lightUbo.iblParams.z, 0.0);
    float maxReflectionLod = max(lightUbo.iblParams.w, 0.0);

    // Fresnel at grazing angle for ambient 
    vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);

    // Diffuse IBL
    vec3 kD_ibl     = (vec3(1.0) - F_ibl) * (1.0 - metalness);
    vec3 irradiance = texture(irradianceMap, normal).rgb;
    vec3 ibl_diffuse = kD_ibl * albedo * irradiance * iblDiffuseScale;

    // Specular IBL
    vec3 R = reflect(-viewDir, normal);
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * maxReflectionLod).rgb;
    vec2 brdf             = texture(brdfLut, vec2(NdotV, roughness)).rg;
    vec3 ibl_specular     = prefilteredColor * (F_ibl * brdf.x + brdf.y) * iblSpecularScale;

    vec3 iblLighting = (ibl_diffuse + ibl_specular) * iblIntensity;

    // Composite
    vec3 finalColor = mix(albedo, iblLighting, lightingMix);

    fragColor = vec4(finalColor, 1.0);
}
