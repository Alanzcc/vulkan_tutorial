struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
}

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
}
ConstantBuffer<UniformBuffer> ubo;

Sampler2D texture;

struct VSOutput {
    float4 sv_position : SV_POSITION;
    float3 color;
    float2 fragTexCoord;
}

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0f))));
    output.color = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    return texture.Sample(vertIn.fragTexCoord);
    // float4(vertIn.fragTexCoord, 0.0f, 1.0f);
}