#include <glm/glm.hpp>
#include <algorithm>
#include <cstdint>
#include <numbers>
#include <print>
#include <ranges>
#include <unordered_map>
#include <vector>

// ─── Core types ──────────────────────────────────────────────────────────────

struct HullFace
{
    uint32_t a, b, c; // indices into the working point cloud
    glm::vec3 normal;
    float offset; // dot(normal, point_on_face)
};

static HullFace makeFace(uint32_t a, uint32_t b, uint32_t c,
                         const std::vector<glm::vec3> &pts)
{
    glm::vec3 n = glm::normalize(glm::cross(pts[b] - pts[a], pts[c] - pts[a]));
    return {a, b, c, n, glm::dot(n, pts[a])};
}

static float signedDist(const HullFace &f, glm::vec3 p)
{
    return glm::dot(f.normal, p) - f.offset;
}

// ─── QuickHull3D ─────────────────────────────────────────────────────────────

std::pair<std::vector<glm::vec3>, std::vector<uint32_t>>
quickHull3D(const std::vector<glm::vec3> &inputPoints)
{
    const uint32_t N = static_cast<uint32_t>(inputPoints.size());
    if (N < 4)
        return {};

    // ── 1. Seed tetrahedron ─────────────────────────────────────────────────
    // Pick 6 extreme points along ±X/Y/Z, then choose 4 maximally spread ones.
    auto extremeIdx = [&](auto proj)
    {
        return static_cast<uint32_t>(
            std::ranges::max_element(std::views::iota(0u, N), {},
                                     [&](uint32_t i)
                                     { return proj(inputPoints[i]); }) -
            std::views::iota(0u, N).begin());
    };

    std::vector<uint32_t> candidates = {
        extremeIdx([](glm::vec3 v)
                   { return v.x; }),
        extremeIdx([](glm::vec3 v)
                   { return -v.x; }),
        extremeIdx([](glm::vec3 v)
                   { return v.y; }),
        extremeIdx([](glm::vec3 v)
                   { return -v.y; }),
        extremeIdx([](glm::vec3 v)
                   { return v.z; }),
        extremeIdx([](glm::vec3 v)
                   { return -v.z; }),
    };
    std::ranges::sort(candidates);
    candidates.erase(std::ranges::unique(candidates).begin(), candidates.end());

    // seed[0..1]: farthest pair
    uint32_t s0 = 0, s1 = 1;
    float maxDist = 0;
    for (auto i : candidates)
        for (auto j : candidates)
            if (float d = glm::distance(inputPoints[i], inputPoints[j]); d > maxDist)
            {
                maxDist = d;
                s0 = i;
                s1 = j;
            }

    // seed[2]: farthest from line s0–s1
    uint32_t s2 = 0;
    maxDist = -1;
    glm::vec3 lineDir = glm::normalize(inputPoints[s1] - inputPoints[s0]);
    for (uint32_t i = 0; i < N; ++i)
    {
        float d = glm::length(glm::cross(inputPoints[i] - inputPoints[s0], lineDir));
        if (d > maxDist)
        {
            maxDist = d;
            s2 = i;
        }
    }

    // seed[3]: farthest from plane s0–s1–s2
    uint32_t s3 = 0;
    maxDist = -1;
    glm::vec3 planeNormal = glm::normalize(
        glm::cross(inputPoints[s1] - inputPoints[s0],
                   inputPoints[s2] - inputPoints[s0]));
    for (uint32_t i = 0; i < N; ++i)
    {
        float d = std::abs(glm::dot(inputPoints[i] - inputPoints[s0], planeNormal));
        if (d > maxDist)
        {
            maxDist = d;
            s3 = i;
        }
    }

    // ── 2. Build initial hull faces (outward-facing) ────────────────────────
    std::vector<HullFace> hull;
    hull.reserve(4);

    auto addFaceOutward = [&](uint32_t a, uint32_t b, uint32_t c,
                              glm::vec3 interior)
    {
        HullFace f = makeFace(a, b, c, inputPoints);
        if (signedDist(f, interior) > 0)
            std::swap(f.b, f.c);                  // flip winding
        f = makeFace(f.a, f.b, f.c, inputPoints); // recompute
        hull.push_back(f);
    };

    glm::vec3 centroid = (inputPoints[s0] + inputPoints[s1] +
                          inputPoints[s2] + inputPoints[s3]) *
                         0.25f;
    addFaceOutward(s0, s1, s2, centroid);
    addFaceOutward(s0, s1, s3, centroid);
    addFaceOutward(s0, s2, s3, centroid);
    addFaceOutward(s1, s2, s3, centroid);

    // ── 3. Assign each point to the farthest visible face ──────────────────
    //    outsideSets[f] = list of (point_idx, signed_distance)
    const float eps = 1e-6f;
    std::vector<std::vector<std::pair<uint32_t, float>>> outsideSets(4);

    for (uint32_t i = 0; i < N; ++i)
    {
        float bestDist = eps;
        int bestFace = -1;
        for (int f = 0; f < (int)hull.size(); ++f)
            if (float d = signedDist(hull[f], inputPoints[i]); d > bestDist)
            {
                bestDist = d;
                bestFace = f;
            }
        if (bestFace >= 0)
            outsideSets[bestFace].emplace_back(i, bestDist);
    }

    // ── 4. Iterative expansion ──────────────────────────────────────────────
    for (int faceIdx = 0; faceIdx < (int)hull.size(); ++faceIdx)
    {
        if (outsideSets[faceIdx].empty())
            continue;

        // Eye point = farthest point from this face
        auto &outside = outsideSets[faceIdx];
        uint32_t eyeIdx = std::ranges::max_element(outside, {},
                                                   [](auto &p)
                                                   { return p.second; })
                              ->first;
        glm::vec3 eye = inputPoints[eyeIdx];

        // Collect all faces visible from eye
        std::vector<bool> visible(hull.size(), false);
        for (int f = 0; f < (int)hull.size(); ++f)
            if (signedDist(hull[f], eye) > eps)
                visible[f] = true;

        // Extract horizon edges (boundary between visible / non-visible)
        using Edge = std::pair<uint32_t, uint32_t>;
        std::vector<Edge> horizonEdges;
        for (int f = 0; f < (int)hull.size(); ++f)
        {
            if (!visible[f])
                continue;
            auto &[a, b, c, n, o] = hull[f];
            for (auto [ea, eb] : {Edge{a, b}, Edge{b, c}, Edge{c, a}})
            {
                // Check if the reverse edge belongs to a non-visible face
                bool onHorizon = std::ranges::any_of(
                    std::views::iota(0, (int)hull.size()),
                    [&](int g)
                    {
                        if (visible[g])
                            return false;
                        auto &[ga, gb, gc, gn, go] = hull[g];
                        for (auto [ha, hb] : {Edge{ga, gb}, Edge{gb, gc}, Edge{gc, ga}})
                            if (ha == eb && hb == ea)
                                return true;
                        return false;
                    });
                if (onHorizon)
                    horizonEdges.emplace_back(ea, eb);
            }
        }

        // Gather all outside points from visible faces
        std::vector<std::pair<uint32_t, float>> orphans;
        for (int f = 0; f < (int)hull.size(); ++f)
            if (visible[f])
            {
                orphans.insert(orphans.end(),
                               outsideSets[f].begin(), outsideSets[f].end());
                outsideSets[f].clear();
            }

        // Remove visible faces (back-to-front to preserve indices)
        for (int f = (int)hull.size() - 1; f >= 0; --f)
            if (visible[f])
            {
                hull.erase(hull.begin() + f);
                outsideSets.erase(outsideSets.begin() + f);
            }

        // Add new faces connecting horizon to eye, reassign orphans
        for (auto [ha, hb] : horizonEdges)
        {
            HullFace newFace = makeFace(ha, hb, eyeIdx, inputPoints);
            // Ensure outward winding
            if (signedDist(newFace, centroid) > 0)
            {
                std::swap(newFace.b, newFace.c);
                newFace = makeFace(newFace.a, newFace.b, newFace.c, inputPoints);
            }
            hull.push_back(newFace);
            outsideSets.emplace_back();

            auto &newOutside = outsideSets.back();
            for (auto &[pi, _] : orphans)
            {
                float d = signedDist(newFace, inputPoints[pi]);
                if (d > eps)
                    newOutside.emplace_back(pi, d);
            }
        }

        // Restart scan from 0 when the hull grew
        faceIdx = -1; // loop increment will make it 0
    }

    // ── 5. Collect unique hull vertices and remap indices ───────────────────
    std::unordered_map<uint32_t, uint32_t> oldToNew;
    std::vector<glm::vec3> hullVertices;
    std::vector<uint32_t> hullIndices;
    hullIndices.reserve(hull.size() * 3);

    auto remap = [&](uint32_t oldIdx) -> uint32_t
    {
        auto [it, inserted] = oldToNew.emplace(oldIdx,
                                               static_cast<uint32_t>(hullVertices.size()));
        if (inserted)
            hullVertices.push_back(inputPoints[oldIdx]);
        return it->second;
    };

    for (auto &[a, b, c, n, o] : hull)
    {
        hullIndices.push_back(remap(a));
        hullIndices.push_back(remap(b));
        hullIndices.push_back(remap(c));
    }

    return {hullVertices, hullIndices};
}

// ─── Test helpers ─────────────────────────────────────────────────────────────

static std::vector<glm::vec3> makeIcosphere(float radius = 1.f)
{
    const float t = (1.f + std::sqrt(5.f)) / 2.f; // golden ratio
    std::vector<glm::vec3> pts;
    for (auto [a, b] : std::initializer_list<std::pair<float, float>>{
             {-1, t}, {1, t}, {-1, -t}, {1, -t}, {
                                                     0,
                                                     -1,
                                                 },
             {0, 1},
             {-t, 0},
             {t, 0}, // (reordered below)
             {0, -1},
             {0, 1},
             {-t, 0},
             {t, 0}})
        (void)a; // generated below

    // 12 icosahedron vertices
    pts = {
        {-1, t, 0},
        {1, t, 0},
        {-1, -t, 0},
        {1, -t, 0},
        {0, -1, t},
        {0, 1, t},
        {0, -1, -t},
        {0, 1, -t},
        {t, 0, -1},
        {t, 0, 1},
        {-t, 0, -1},
        {-t, 0, 1},
    };
    for (auto &p : pts)
        p = glm::normalize(p) * radius;
    return pts;
}

static std::vector<glm::vec3> makeCube(float half = 1.f)
{
    std::vector<glm::vec3> pts;
    for (int x : {-1, 1})
        for (int y : {-1, 1})
            for (int z : {-1, 1})
                pts.push_back({x * half, y * half, z * half});
    return pts;
}

static std::vector<glm::vec3> makeTetrahedron(float r = 1.f)
{
    return {
        glm::normalize(glm::vec3{1, 1, 1}) * r,
        glm::normalize(glm::vec3{1, -1, -1}) * r,
        glm::normalize(glm::vec3{-1, 1, -1}) * r,
        glm::normalize(glm::vec3{-1, -1, 1}) * r,
    };
}

static std::vector<glm::vec3> makeOctahedron(float r = 1.f)
{
    return {
        {r, 0, 0},
        {-r, 0, 0},
        {0, r, 0},
        {0, -r, 0},
        {0, 0, r},
        {0, 0, -r},
    };
}

static void runTest(const std::string &name,
                    const std::vector<glm::vec3> &pts,
                    uint32_t expectedVertices,
                    uint32_t expectedTriangles)
{
    auto [verts, indices] = quickHull3D(pts);
    bool vertOk = verts.size() == expectedVertices;
    bool triOk = indices.size() == expectedTriangles * 3;
    std::println("{:20s} verts={:2d}/{:2d} {} | tris={:2d}/{:2d} {}",
                 name,
                 verts.size(), expectedVertices, vertOk ? "✓" : "✗",
                 indices.size() / 3, expectedTriangles, triOk ? "✓" : "✗");
}

// ─── main ─────────────────────────────────────────────────────────────────────
/*
int main()
{
    std::println("=== QuickHull3D Tests ===\n");

    // Icosphere: 12 hull verts, 20 triangular faces
    runTest("Icosphere",    makeIcosphere(), 12, 20);

    // Cube: 8 hull verts, 6 quad-faces = 12 triangles
    runTest("Cube",         makeCube(),       8, 12);

    // Tetrahedron: 4 hull verts, 4 triangular faces
    runTest("Tetrahedron",  makeTetrahedron(), 4,  4);

    // Octahedron: 6 hull verts, 8 triangular faces
    runTest("Octahedron",   makeOctahedron(),  6,  8);

    // ── Robustness: add interior noise — hull counts must not change ─────────
    //
    // Strategy: any convex combination  Σ wᵢ·vᵢ  (wᵢ ≥ 0, Σwᵢ = 1)  of the
    // hull vertices is guaranteed to lie inside the convex hull.
    // We build each random point as a weighted blend of all hull vertices where
    // the weights are random non-negative values normalised to sum to 1.
    // This works for every convex shape regardless of its geometry.


    std::println("\n-- With 100 interior noise points --");
    runTest("Icosphere+noise",   withInteriorNoise(makeIcosphere()),   12, 20);
    runTest("Cube+noise",        withInteriorNoise(makeCube()),          8, 12);
    runTest("Tetrahedron+noise", withInteriorNoise(makeTetrahedron()),   4,  4);
    runTest("Octahedron+noise",  withInteriorNoise(makeOctahedron()),    6,  8);
}
    */