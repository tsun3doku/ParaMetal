const float PI = 3.14159265359;
const float WALK_AREA_EPSILON = 1e-10;
const int WALK_MAX_STEPS = 256;
const int WALK_SUCCESS = 0;
const int WALK_NO_SEED = -1;
const int WALK_INVALID_TOPOLOGY = -2;
const int WALK_DEGENERATE = -3;
const int WALK_BOUNDARY = -4;
const int WALK_LIMIT_REACHED = -5;
const int WALK_STALLED = -6;

struct WalkResult {
    int triangle;
    int status;
    int steps;
    vec3 bary;
    vec3 baryDx;
    vec3 baryDy;
};

WalkResult makeWalkResult(int status) {
    WalkResult result;
    result.triangle = -1;
    result.status = status;
    result.steps = 0;
    result.bary = vec3(1.0, 0.0, 0.0);
    result.baryDx = result.bary;
    result.baryDy = result.bary;
    return result;
}

bool validIndex(int index, int count) {
    return index >= 0 && index < count;
}

bool validLength(float lengthValue) {
    return lengthValue > 0.0 && !isnan(lengthValue) && !isinf(lengthValue);
}

bool ccw(vec2 p, vec2 q, vec2 r) {
    float det = (p.x - r.x) * (q.y - r.y) - (p.y - r.y) * (q.x - r.x);
    return det >= 0.0;
}

bool crossing(vec2 origin, vec2 v_a, vec2 v_b, vec2 p) {
    bool t0 = ccw(origin, p, v_a);
    bool t1 = ccw(origin, p, v_b);
    bool u0 = ccw(v_a, v_b, origin);
    bool u1 = ccw(v_a, v_b, p);
    return ((t0 && !t1) || (!t0 && t1)) && ((u0 && !u1) || (!u0 && u1));
}

float area(vec2 p, vec2 q, vec2 r) {
    float det = (q.x * r.y + p.x * q.y + p.y * r.x) - (q.x * p.y + r.x * q.y + r.y * p.x);
    return det / 2.0;
}

int mate(int he, int halfedgeCount, int edgeCount) {
    if (!validIndex(he, halfedgeCount)) return -2;

    int edgeIdx = texelFetch(H, he).g;
    if (!validIndex(edgeIdx, edgeCount)) return -2;

    ivec2 edge = texelFetch(E, edgeIdx).rg;
    int opposite = -2;
    if (edge.r == he) opposite = edge.g;
    else if (edge.g == he) opposite = edge.r;

    if (opposite == -1) return -1;
    if (!validIndex(opposite, halfedgeCount) || opposite == he) return -2;
    return opposite;
}

bool loadInputTriangleChart(int inputTri, out ivec3 faceHEs, out vec2 triCoords[3]) {
    int inputTriangleCount = textureSize(T_input);
    int inputHalfedgeCount = textureSize(H_input);
    if (!validIndex(inputTri, inputTriangleCount)) return false;

    int h0 = texelFetch(T_input, inputTri).r;
    if (!validIndex(h0, inputHalfedgeCount)) return false;

    int h1 = texelFetch(H_input, h0).a;
    if (!validIndex(h1, inputHalfedgeCount)) return false;

    int h2 = texelFetch(H_input, h1).a;
    if (!validIndex(h2, inputHalfedgeCount) || texelFetch(H_input, h2).a != h0) return false;

    faceHEs = ivec3(h0, h1, h2);

    int e0 = texelFetch(H_input, h0).g;
    int e1 = texelFetch(H_input, h1).g;
    int e2 = texelFetch(H_input, h2).g;
    int inputEdgeCount = textureSize(L_input);
    if (!validIndex(e0, inputEdgeCount) || !validIndex(e1, inputEdgeCount) || !validIndex(e2, inputEdgeCount)) return false;

    float a = texelFetch(L_input, e0).r;
    float b = texelFetch(L_input, e1).r;
    float c = texelFetch(L_input, e2).r;
    if (!validLength(a) || !validLength(b) || !validLength(c)) return false;

    triCoords[0] = vec2(0.0, 0.0);
    triCoords[1] = vec2(a, 0.0);
    float x = (a * a + c * c - b * b) / (2.0 * a);
    float y2 = c * c - x * x;
    if (!(y2 > WALK_AREA_EPSILON) || isnan(y2) || isinf(y2)) return false;
    triCoords[2] = vec2(x, sqrt(y2));
    return true;
}

vec2 inputChartPoint(vec2 triCoords[3], vec3 inputBary) {
    return inputBary.x * triCoords[0] + inputBary.y * triCoords[1] + inputBary.z * triCoords[2];
}

bool remapBarycentric(int href, int h0, int h1, int h2, vec3 localBary, out vec3 bary) {
    if (href == h0) bary = localBary;
    else if (href == h1) bary = localBary.yzx;
    else if (href == h2) bary = localBary.zxy;
    else return false;
    return true;
}

bool intrinsicTriangleHalfedges(int triangle, out ivec3 halfedges) {
    int triangleCount = textureSize(T);
    if (!validIndex(triangle, triangleCount)) {
        return false;
    }

    int href = texelFetch(T, triangle).r;
    int halfedgeCount = textureSize(H);
    if (!validIndex(href, halfedgeCount)) {
        return false;
    }

    int h1 = texelFetch(H, href).a;
    if (!validIndex(h1, halfedgeCount)) {
        return false;
    }

    int h2 = texelFetch(H, h1).a;
    if (!validIndex(h2, halfedgeCount) ||
        texelFetch(H, h2).a != href) {
        return false;
    }

    halfedges = ivec3(href, h1, h2);
    return true;
}

bool intrinsicTriangleEdgeIds(int triangle, out uvec3 edgeIds) {
    ivec3 halfedges;
    if (!intrinsicTriangleHalfedges(triangle, halfedges)) {
        return false;
    }

    int edgeCount = textureSize(E);
    int e0 = texelFetch(H, halfedges.y).g;
    int e1 = texelFetch(H, halfedges.z).g;
    int e2 = texelFetch(H, halfedges.x).g;
    if (!validIndex(e0, edgeCount) ||
        !validIndex(e1, edgeCount) ||
        !validIndex(e2, edgeCount)) {
        return false;
    }

    edgeIds = uvec3(uint(e0), uint(e1), uint(e2));
    return true;
}

bool intrinsicTriangleVertices(int triangle, out ivec3 vertices) {
    ivec3 halfedges;
    if (!intrinsicTriangleHalfedges(triangle, halfedges)) {
        return false;
    }

    vertices = ivec3(
        texelFetch(H, halfedges.x).r,
        texelFetch(H, halfedges.y).r,
        texelFetch(H, halfedges.z).r);
    return true;
}

float intrinsicEdgeDistance(float bary, vec2 gradient) {
    const float invalidDistance = uintBitsToFloat(0x7f800000u);
    float lengthSquared = dot(gradient, gradient);
    if (!(lengthSquared > 0.0) ||
        isnan(lengthSquared) ||
        isinf(lengthSquared)) {
        return invalidDistance;
    }
    return abs(bary) * inversesqrt(lengthSquared);
}

vec3 intrinsicEdgeDistances(WalkResult walk) {
    vec3 dbdx = walk.baryDx - walk.bary;
    vec3 dbdy = walk.baryDy - walk.bary;
    return vec3(
        intrinsicEdgeDistance(walk.bary.x, vec2(dbdx.x, dbdy.x)),
        intrinsicEdgeDistance(walk.bary.y, vec2(dbdx.y, dbdy.y)),
        intrinsicEdgeDistance(walk.bary.z, vec2(dbdx.z, dbdy.z)));
}

WalkResult walkFromSeed(vec2 p, vec2 px, vec2 py, int inputHe, vec2 seedOrigin, float seedPhi, bool calculateDerivatives) {
    WalkResult result = makeWalkResult(WALK_INVALID_TOPOLOGY);
    int supportCount = textureSize(S);
    int angleCount = textureSize(A);
    int halfedgeCount = textureSize(H);
    int edgeCount = textureSize(E);
    int triangleCount = textureSize(T);
    int lengthCount = textureSize(L);
    if (!validIndex(inputHe, supportCount) || !validIndex(inputHe, angleCount)) return result;

    int h0 = texelFetch(S, inputHe).r;
    if (h0 < 0) {
        result.status = WALK_NO_SEED;
        return result;
    }
    if (!validIndex(h0, halfedgeCount)) return result;

    float phi0 = seedPhi + texelFetch(A, inputHe).r;
    ivec4 he0 = texelFetch(H, h0);
    int e0 = he0.g;
    if (!validIndex(e0, lengthCount)) return result;

    float l0 = texelFetch(L, e0).r;
    if (!validLength(l0) || isnan(phi0) || isinf(phi0)) return result;

    vec2 v0 = seedOrigin;
    vec2 v1 = v0 + l0 * vec2(cos(phi0), sin(phi0));
    for (int iter = 0; iter < WALK_MAX_STEPS; ++iter) {
        result.steps = iter + 1;
        if (!validIndex(h0, halfedgeCount)) return result;

        ivec4 halfedge = texelFetch(H, h0);
        int t = halfedge.b;
        int h1 = halfedge.a;
        if (!validIndex(t, triangleCount) || !validIndex(h1, halfedgeCount)) return result;

        ivec4 he1 = texelFetch(H, h1);
        int h2 = he1.a;
        if (!validIndex(h2, halfedgeCount)) return result;

        ivec4 he2 = texelFetch(H, h2);
        if (he2.a != h0) return result;

        int e1 = he1.g;
        int e2 = he2.g;
        if (!validIndex(e1, lengthCount) || !validIndex(e2, lengthCount)) return result;

        float l1 = texelFetch(L, e1).r;
        float l2 = texelFetch(L, e2).r;
        if (!validLength(l1) || !validLength(l2)) return result;

        float alphaDenominator = 2.0 * l0 * l1;
        if (!validLength(alphaDenominator)) return result;
        float alpha = acos(clamp((l0 * l0 + l1 * l1 - l2 * l2) / alphaDenominator, -1.0, 1.0));
        float phi1 = phi0 + PI - alpha;
        vec2 v2 = v1 + l1 * vec2(cos(phi1), sin(phi1));
        float triangleArea = area(v0, v1, v2);
        if (isnan(triangleArea) || isinf(triangleArea) || abs(triangleArea) <= WALK_AREA_EPSILON) {
            result.status = WALK_DEGENERATE;
            return result;
        }

        bool outsideEdge12 = !ccw(v1, v2, p);
        bool outsideEdge20 = !ccw(v2, v0, p);
        if (outsideEdge12 && (ccw(v2, v0, p) || crossing(seedOrigin, v1, v2, p))) {
            int nextH = mate(h1, halfedgeCount, edgeCount);
            if (nextH == -1) {
                result.status = WALK_BOUNDARY;
                return result;
            }
            if (nextH < 0) return result;

            v0 = v2;
            l0 = l1;
            h0 = nextH;
            phi0 = phi1 + PI;
            continue;
        }

        if (outsideEdge20) {
            int nextH = mate(h2, halfedgeCount, edgeCount);
            if (nextH == -1) {
                result.status = WALK_BOUNDARY;
                return result;
            }
            if (nextH < 0) return result;

            float betaDenominator = 2.0 * l0 * l2;
            if (!validLength(betaDenominator)) return result;
            float beta = acos(clamp((l0 * l0 + l2 * l2 - l1 * l1) / betaDenominator, -1.0, 1.0));

            v1 = v2;
            l0 = l2;
            h0 = nextH;
            phi0 += beta;
            continue;
        }

        vec3 localBary = vec3(
            area(v1, v2, p) / triangleArea,
            area(v2, v0, p) / triangleArea,
            0.0);
        localBary.z = 1.0 - localBary.x - localBary.y;

        int href = texelFetch(T, t).r;
        if (!remapBarycentric(href, h0, h1, h2, localBary, result.bary)) return result;

        if (calculateDerivatives) {
            vec3 localBaryX = vec3(
                area(v1, v2, px) / triangleArea,
                area(v2, v0, px) / triangleArea,
                0.0);
            localBaryX.z = 1.0 - localBaryX.x - localBaryX.y;

            vec3 localBaryY = vec3(
                area(v1, v2, py) / triangleArea,
                area(v2, v0, py) / triangleArea,
                0.0);
            localBaryY.z = 1.0 - localBaryY.x - localBaryY.y;

            if (!remapBarycentric(href, h0, h1, h2, localBaryX, result.baryDx) ||
                !remapBarycentric(href, h0, h1, h2, localBaryY, result.baryDy)) return result;
        } else {
            result.baryDx = result.bary;
            result.baryDy = result.bary;
        }

        result.triangle = t;
        result.status = WALK_SUCCESS;
        return result;
    }

    result.status = WALK_LIMIT_REACHED;
    return result;
}

WalkResult locateIntrinsicTriangle(ivec3 faceHEs, vec2 triCoords[3], vec2 p, vec2 px, vec2 py, bool calculateDerivatives) {
    int inputHe = faceHEs[0];
    if (!validIndex(inputHe, textureSize(S)) || texelFetch(S, inputHe).r < 0) {
        return makeWalkResult(WALK_NO_SEED);
    }

    vec2 edgeDir = triCoords[1] - triCoords[0];
    return walkFromSeed(p, px, py, inputHe, triCoords[0], atan(edgeDir.y, edgeDir.x), calculateDerivatives);
}
