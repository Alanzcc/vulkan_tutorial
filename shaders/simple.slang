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

struct InstanceLUT {
    uint materialID;
    uint indexBufferOffset;
  };
[[vk::binding(4,0)]];
StructuredBuffer<InstanceLUT> instanceLUTBuffer;

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

[[vk::binding(0,1)]]
SamplerState textureSampler;

[[vk::binding(1,1)]]
Texture2D<float4> textures[];

struct PushConstant {
    uint materialIndex;
}
[[push_constant]]
PushConstant pc;


static const float3 lightDir = float3(-6.0, 0.0, 6.0);
static const float EPSILON = 0.01;

bool in_shadow(float3 P) {
   // Build the shadow ray from the world position toward the light
   RayDesc shadowRayDesc;
   shadowRayDesc.Origin = P;
   shadowRayDesc.Direction = normalize(lightDir);
   shadowRayDesc.TMin = EPSILON;
   shadowRayDesc.TMax = 1e4;

   RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
   let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

   sq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, shadowRayDesc);
   sq.Proceed();

   // If the shadow ray intersects an opaque triangle, we consider the pixel in shadow
   bool hit = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

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
  
  return baseColor;
}
