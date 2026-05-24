struct VSInput {
    float2 inPosition;
    float3 inColor;
}

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
}
ConstantBuffer<UniformBuffer> ubo;

struct VSOutput {
    float4 sv_position : SV_POSITION;
    float3 color;
}

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 0.0f, 1.0f))));
    output.color = input.inColor;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target {
    return float4(vertIn.color, 1.0f);
}