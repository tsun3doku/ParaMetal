#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

// Inputs from vertex shader
layout(location = 0) in vec3 inPosition[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec3 inColor[];
layout(location = 3) in vec2 inTexCoord[];

// Outputs to fragment shader
layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) out vec3 fragBaryCoord;
layout(location = 5) out vec2 fragIntrinsicCoord;

void main() {
    // Pass through gl_PrimitiveID
    gl_PrimitiveID = gl_PrimitiveIDIn;
    
    // Emit vertex 0
    gl_Position = gl_in[0].gl_Position;
    fragPosition = inPosition[0];
    fragNormal = inNormal[0];
    fragColor = inColor[0];
    fragTexCoord = inTexCoord[0];
    fragBaryCoord = vec3(1.0, 0.0, 0.0);
    fragIntrinsicCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Emit vertex 1
    gl_Position = gl_in[1].gl_Position;
    fragPosition = inPosition[1];
    fragNormal = inNormal[1];
    fragColor = inColor[1];
    fragTexCoord = inTexCoord[1];
    fragBaryCoord = vec3(0.0, 1.0, 0.0);
    fragIntrinsicCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Emit vertex 2
    gl_Position = gl_in[2].gl_Position;
    fragPosition = inPosition[2];
    fragNormal = inNormal[2];
    fragColor = inColor[2];
    fragTexCoord = inTexCoord[2];
    fragBaryCoord = vec3(0.0, 0.0, 1.0);
    fragIntrinsicCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    EndPrimitive();
}
