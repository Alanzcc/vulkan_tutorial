struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float3 cameraPos;
};

[[vk::binding(0,0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(1,0)]]
RaytracingAccelerationStructure accelerationStructure;

[[vk::binding(2,0)]]
StructuredBuffer<uint> indexBuffer;

[[vk::binding(3,0)]]
StructuredBuffer<float2> uvBuffer;

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 worldPos;
}

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0f))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.fragNormal = input.inNormal;
    output.worldPos = mul(ubo.model, float4(input.inPosition, 1.0)).xyz;
    return output;
}

bool in_shadow(float3 P) {
    bool hit = false;
    
    return hit;
  }

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
  float4 baseColor = textures[pc.materialIndex].Sample(textureSampler, vertIn.fragTexCoord);

  float3 P = vertIn.worldPos;

  bool inShadow = in_shadow(P);

  float4(vertIn.fragTexCoord, 0.0f, 1.0f);

  // Darken if in shadows
  if (inShadow) {
      baseColor.rgb *= 0.2;
  }




}
