static const uint InvalidSeed = 0xFFFFFFFFu;
static const uint SolidSeed   = 0xFFFFFFFEu;
static const float InvalidTemperature = -1e30f;
static const float SolidTemperature   = -2e30f;

[[vk::binding(0, 0)]] StructuredBuffer<uint> fluidSeedGrid;
[[vk::binding(1, 0)]] StructuredBuffer<uint> seedToFluidNode;
[[vk::binding(2, 0)]] StructuredBuffer<float> temperatureSource;
[[vk::binding(3, 0)]] RWTexture3D<float> temperatureVolume;

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    uint3 dim;
    temperatureVolume.GetDimensions(dim.x, dim.y, dim.z);
    if (any(id >= dim)) return;

    uint index = id.x + id.y * dim.x + id.z * dim.x * dim.y;
    uint seed = fluidSeedGrid[index];

    if (seed == SolidSeed) {
        temperatureVolume[id] = SolidTemperature;
        return;
    }
    if (seed == InvalidSeed) {
        temperatureVolume[id] = InvalidTemperature;
        return;
    }

    uint node = seedToFluidNode[seed];
    temperatureVolume[id] = (node != 0xFFFFFFFFu)
        ? temperatureSource[node]
        : InvalidTemperature;
}
