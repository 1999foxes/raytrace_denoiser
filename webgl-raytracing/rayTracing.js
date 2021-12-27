/*
 WebGL Path Tracing With Temporal Spatial Denoiser

 License: MIT License (see below)

 Copyright (c) 2010 Evan Wallace

 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
*/

////////////////////////////////////////////////////////////////////////////////
// shader strings
////////////////////////////////////////////////////////////////////////////////

// vertex shader for denoiser, which draws a textured quad
var denoiserVertexSource = 
`#version 300 es
in vec3 vertex; 
out vec2 texCoord; 
void main() {
    texCoord = vertex.xy * 0.5 + 0.5;
    gl_Position = vec4(vertex, 1.0);
}`;

// fragment shader for temporal denoiser
var temporalDenoiserFragmentSource =
`#version 300 es
precision highp float;
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 colorSquare;
in vec2 texCoord; 
uniform sampler2D colorTexture;
uniform sampler2D gBufferTexture;
uniform sampler2D integratedColorTexture;
uniform sampler2D integratedColorSquareTexture;
uniform sampler2D lastGBufferTexture;


vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}


void main() {
    vec4 rawColor = texture(colorTexture, texCoord);
    vec4 rawColorSquare = rawColor * rawColor;
    vec4 gBuffer = texture(gBufferTexture, texCoord);
    vec4 integratedColor = texture(integratedColorTexture, texCoord);
    vec4 integratedColorSquare = texture(integratedColorSquareTexture, texCoord);
    vec4 lastGBuffer = texture(lastGBufferTexture, texCoord);

    vec4 varianceVec = vec4((integratedColorSquare - integratedColor * integratedColor).xyz, 0.0);
    float variance = dot(varianceVec, varianceVec);
    float sampleConfidence = integratedColorSquare.w;

    float deltaDepth = abs(lastGBuffer.w - gBuffer.w);
    
    vec3 deltaNormal = gBuffer.xyz - lastGBuffer.xyz;
    float dotDeltaNormal = dot(deltaNormal, deltaNormal);

    vec3 clampColorMin = vec3(1,1,1);
    vec3 clampColorMax = vec3(0,0,0);
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        vec3 neighbor = rgb2hsv(texture(colorTexture, texCoord + vec2((1.0 / 512.0 * float(i - 2)), (1.0 / 512.0 * float(j - 2)))).xyz);
        clampColorMin = vec3(min(clampColorMin.x, neighbor.x), min(clampColorMin.y, neighbor.y), min(clampColorMin.z, neighbor.z));
        clampColorMax = vec3(max(clampColorMax.x, neighbor.x), max(clampColorMax.y, neighbor.y), max(clampColorMax.z, neighbor.z));
      }
    }

    float mixWeight = 0.0;
    vec3 hsvIntegratedColor = rgb2hsv(integratedColor.xyz);
    if (deltaDepth < 0.01 && dotDeltaNormal < 0.01 
        && lessThanEqual(clampColorMin, hsvIntegratedColor) == bvec3(true,true,true) 
        && greaterThanEqual(clampColorMax, hsvIntegratedColor) == bvec3(true,true,true)) {
      mixWeight = 0.8;
      sampleConfidence += 0.1;
    } else {
      // disocclusion
      mixWeight = 0.0;
      sampleConfidence = 0.0;
    }

    fragColor = mix(rawColor, integratedColor, mixWeight);
    fragColor.w = 1.0;

    colorSquare = mix(rawColorSquare, integratedColorSquare, mixWeight);
    colorSquare.w = sampleConfidence;  
}`;


// fragment shader for spatial denoiser
var spatialDenoiserFragmentSource =
`#version 300 es
precision highp float;
layout(location = 0) out vec4 fragColor;
in vec2 texCoord;
uniform sampler2D integratedColorTexture;
uniform sampler2D gBufferTexture;
uniform sampler2D integratedColorSquareTexture;
uniform int stepLength;

void main() {
    vec4 color = texture(integratedColorTexture, texCoord);
    vec4 colorSquare = texture(integratedColorSquareTexture, texCoord);
    vec4 gBuffer = texture(gBufferTexture, texCoord);

    vec3 normal = (gBuffer.xyz - vec3(0.5, 0.5, 0.5)) * 2.0;
    float depth = gBuffer.w;
    vec2 grad = vec2(dFdx(depth), dFdy(depth));
    
    vec4 colorSum = vec4(0,0,0,0);
    float weightSum = 0.0;
    float varianceSum = 0.0;
    float sampleConfidenceSum = 0.0;
    float kernal[25] = float[](0.01247764154323288, 0.02641516735431067, 0.03391774626899505, 0.02641516735431067, 0.01247764154323288, 
      0.02641516735431067, 0.05592090972790157, 0.07180386941492609, 0.05592090972790157, 0.02641516735431067, 
      0.03391774626899505, 0.07180386941492609, 0.09219799334529226, 0.07180386941492609, 0.03391774626899505, 
      0.02641516735431067, 0.05592090972790157, 0.07180386941492609, 0.05592090972790157, 0.02641516735431067, 
      0.01247764154323288, 0.02641516735431067, 0.03391774626899505, 0.02641516735431067, 0.01247764154323288);
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        int index = i * 5 + j;

        vec4 color2 = texture(integratedColorTexture, texCoord + vec2((1.0 / 512.0 * float((i - 2) * stepLength)), (1.0 / 512.0 * float((j - 2) * stepLength))));
        vec4 colorSquare2 = texture(integratedColorSquareTexture, texCoord + vec2((1.0 / 512.0 * float((i - 2) * stepLength)), (1.0 / 512.0 * float((j - 2) * stepLength))));
        vec4 gBuffer2 = texture(gBufferTexture, texCoord + vec2((1.0 / 512.0 * float((i - 2) * stepLength)), (1.0 / 512.0 * float((j - 2) * stepLength))));

        vec4 varianceVec2 = vec4((colorSquare2 - color2 * color2).xyz, 0.0);
        float variance2 = dot(varianceVec2, varianceVec2);
        float sampleConfidence2 = colorSquare2.w;

        vec3 normal2 = (gBuffer2.xyz - vec3(0.5, 0.5, 0.5)) * 2.0;
        float depth2 = gBuffer2.w;

        float weightNormal = pow(max(0.0, dot(normal, normal2)), 64.0);
        float weightDepth = exp((-abs(depth - depth2)) / (abs(dot(grad, vec2(i-2, j-2))) + 0.01));
        float weight = weightNormal * weightDepth * kernal[index];

        weightSum += weight;
        colorSum += weight * color2;
        varianceSum += weight * variance2;
        sampleConfidenceSum += weight * sampleConfidence2;
      }
    }
    colorSum /= weightSum; 
    varianceSum /= weightSum; 
    sampleConfidenceSum /= weightSum; 

    float variance = clamp(max(varianceSum * 50.0, 1.0 - sampleConfidenceSum), 0.2, 0.8);
    fragColor = mix(color, colorSum, variance);

    fragColor.w = 1.0;
}`;


// vertex shader for drawing a line
var lineVertexSource =
' attribute vec3 vertex;' +
' uniform vec3 cubeMin;' +
' uniform vec3 cubeMax;' +
' uniform mat4 modelviewProjection;' +
' void main() {' +
'   gl_Position = modelviewProjection * vec4(mix(cubeMin, cubeMax, vertex), 1.0);' +
' }';

// fragment shader for drawing a line
var lineFragmentSource =
' precision highp float;' +
' void main() {' +
'   gl_FragColor = vec4(1.0);' +
' }';


// constants for the shaders
var bounces = '5';
var epsilon = '0.0001';
var infinity = '10000.0';
var lightSize = 0.1;
var lightVal = 0.5;

// vertex shader, interpolate ray per-pixel
var tracerVertexSource =
`#version 300 es
in vec3 vertex;
uniform vec3 eye, ray00, ray01, ray10, ray11;
out vec3 initialRay;
void main() {
    vec2 percent = vertex.xy * 0.5 + 0.5;
    initialRay = mix(mix(ray00, ray01, percent.y), mix(ray10, ray11, percent.y), percent.x);
    gl_Position = vec4(vertex, 1.0);
}`;


// fragment shader

// compute specular lighting contribution
var specularReflection =
' vec3 reflectedLight = normalize(reflect(light - hit, normal));' +
' specularHighlight = max(0.0, dot(reflectedLight, normalize(hit - origin)));';

// update ray using normal and bounce according to a diffuse reflection
var newDiffuseRay =
' ray = cosineWeightedDirection(timeSinceStart + float(bounce), normal);';

// update ray using normal according to a specular reflection
var newReflectiveRay =
' ray = reflect(ray, normal);' +
  specularReflection +
' specularHighlight = 2.0 * pow(specularHighlight, 20.0);';

// update ray using normal and bounce according to a glossy reflection
var newGlossyRay =
' ray = normalize(reflect(ray, normal)) + uniformlyRandomVector(timeSinceStart + float(bounce)) * glossiness;' +
  specularReflection +
' specularHighlight = pow(specularHighlight, 3.0);';

var yellowBlueCornellBox =
' if(hit.x < -0.9999) surfaceColor = vec3(0.1, 0.5, 1.0);' + // blue
' else if(hit.x > 0.9999) surfaceColor = vec3(1.0, 0.9, 0.1);'; // yellow

var redGreenCornellBox =
' if(hit.x < -0.9999) surfaceColor = vec3(1.0, 0.3, 0.1);' + // red
' else if(hit.x > 0.9999) surfaceColor = vec3(0.3, 1.0, 0.1);'; // green

function makeShadow(objects) {
  return '' +
' float shadow(vec3 origin, vec3 ray) {' +
    concat(objects, function(o){ return o.getShadowTestCode(); }) +
'   return 1.0;' +
' }';
}

function makeCalculateColor(objects) {
  return '' +
' void calculateColor(vec3 origin, vec3 ray, vec3 light, inout vec3 ioColor, inout vec3 ioNormal, inout float ioDepth) {' +
'   vec3 colorMask = vec3(1.0);' +
'   vec3 accumulatedColor = vec3(0.0);' +
  
    // main raytracing loop
'   for(int bounce = 0; bounce < ' + bounces + '; bounce++) {' +
      // compute the intersection with everything
'     vec2 tRoom = intersectCube(origin, ray, roomCubeMin, roomCubeMax);' +
      concat(objects, function(o){ return o.getIntersectCode(); }) +

      // find the closest intersection
'     float t = ' + infinity + ';' +
'     if (tRoom.x < tRoom.y) t = tRoom.y;' +
      concat(objects, function(o){ return o.getMinimumIntersectCode(); }) +

      // info about hit
'     vec3 hit = origin + ray * t;' +
'     vec3 surfaceColor = vec3(0.75);' +
'     float specularHighlight = 0.0;' +
'     vec3 normal;' +

      // calculate the normal (and change wall color)
'     if(t == tRoom.y) {' +
'       normal = -normalForCube(hit, roomCubeMin, roomCubeMax);' +
        [yellowBlueCornellBox, redGreenCornellBox][environment] +
        newDiffuseRay +
'     } else if(t == ' + infinity + ') {' +
'       break;' +
'     } else {' +
'       if(false) ;' + // hack to discard the first 'else' in 'else if'
        concat(objects, function(o){ return o.getNormalCalculationCode(); }) +
        [newDiffuseRay, newReflectiveRay, newGlossyRay][material] +
'     }' +

      // compute diffuse lighting contribution
'     vec3 toLight = light - hit;' +
'     float diffuse = max(0.0, dot(normalize(toLight), normal));' +

      // trace a shadow ray to the light
'     float shadowIntensity = shadow(hit + normal * ' + epsilon + ', toLight);' +

      // do light bounce
'     colorMask *= surfaceColor;' +
'     accumulatedColor += colorMask * (' + lightVal + ' * diffuse * shadowIntensity);' +
'     accumulatedColor += colorMask * specularHighlight * shadowIntensity;' +

      // calculate next origin
'     origin = hit;' +
      
      // output geometry info
      `if (bounce == 0) {
          ioNormal = normal * 0.5 + 0.5;
          ioDepth = t * 0.05;
      }
      ` +

'   }' +
'   ioColor = accumulatedColor;' +
' }';
}

function makeMain() {
  return `void main() {
    vec3 newLight = light + uniformlyRandomVector(timeSinceStart - 53.0) * ${lightSize};
    vec2 texCoord = gl_FragCoord.xy / 512.0;
    vec3 color; vec3 normal; float depth;
    calculateColor(eye, initialRay, newLight, color, normal, depth);
    fragColor = vec4(color, 1.0);
    colorSquare = vec4(color * color, 1.0);
    gBuffer = vec4(normal, depth);
}`;
}

function makeTracerFragmentSource(objects) {
  const fs = 
  // variables
  `#version 300 es
  precision highp float;
  layout(location = 0) out vec4 fragColor;
  layout(location = 1) out vec4 gBuffer;
  layout(location = 2) out vec4 colorSquare;
  uniform vec3 eye;
  in vec3 initialRay;
  uniform float textureWeight;
  uniform float timeSinceStart;
  uniform float glossiness;
  vec3 roomCubeMin = vec3(-1.0, -1.0, -1.0);
  vec3 roomCubeMax = vec3(1.0, 1.0, 1.0);
  ` + 
  // objects
  concat(objects, function(o){ return o.getGlobalCode(); }) + 
  // path tracing functions
  `vec2 intersectCube(vec3 origin, vec3 ray, vec3 cubeMin, vec3 cubeMax) {
      vec3 tMin = (cubeMin - origin) / ray;
      vec3 tMax = (cubeMax - origin) / ray;
      vec3 t1 = min(tMin, tMax);
      vec3 t2 = max(tMin, tMax);
      float tNear = max(max(t1.x, t1.y), t1.z);
      float tFar = min(min(t2.x, t2.y), t2.z);
      return vec2(tNear, tFar);
  }


  vec3 normalForCube(vec3 hit, vec3 cubeMin, vec3 cubeMax) {
      if(hit.x < cubeMin.x + 0.0001) return vec3(-1.0, 0.0, 0.0);
      else if(hit.x > cubeMax.x - 0.0001) return vec3(1.0, 0.0, 0.0);
      else if(hit.y < cubeMin.y + 0.0001) return vec3(0.0, -1.0, 0.0);
      else if(hit.y > cubeMax.y - 0.0001) return vec3(0.0, 1.0, 0.0);
      else if(hit.z < cubeMin.z + 0.0001) return vec3(0.0, 0.0, -1.0);
      else return vec3(0.0, 0.0, 1.0);
  }


  float intersectSphere(vec3 origin, vec3 ray, vec3 sphereCenter, float sphereRadius) {
      vec3 toSphere = origin - sphereCenter;
      float a = dot(ray, ray);
      float b = 2.0 * dot(toSphere, ray);
      float c = dot(toSphere, toSphere) - sphereRadius*sphereRadius;
      float discriminant = b*b - 4.0*a*c;
      if (discriminant > 0.0) {
          float t = (-b - sqrt(discriminant)) / (2.0 * a);
          if (t > 0.0) return t;
      }
      return 10000.0;
  }


  vec3 normalForSphere(vec3 hit, vec3 sphereCenter, float sphereRadius) {
      return (hit - sphereCenter) / sphereRadius;
  }


  float random(vec3 scale, float seed) {
      return fract(sin(dot(gl_FragCoord.xyz + seed, scale)) * 43758.5453 + seed);
  }


  vec3 cosineWeightedDirection(float seed, vec3 normal) {
      float u = random(vec3(12.9898, 78.233, 151.7182), seed);
      float v = random(vec3(63.7264, 10.873, 623.6736), seed);
      float r = sqrt(u);
      float angle = 6.283185307179586 * v;
      vec3 sdir, tdir;
      if (abs(normal.x)<.5) {
          sdir = cross(normal, vec3(1,0,0));
      } else {
          sdir = cross(normal, vec3(0,1,0));
      }
      tdir = cross(normal, sdir);
      return r*cos(angle)*sdir + r*sin(angle)*tdir + sqrt(1.-u)*normal;
  }


  vec3 uniformlyRandomDirection(float seed) {
      float u = random(vec3(12.9898, 78.233, 151.7182), seed);
      float v = random(vec3(63.7264, 10.873, 623.6736), seed);
      float z = 1.0 - 2.0 * u;
      float r = sqrt(1.0 - z * z);
      float angle = 6.283185307179586 * v;
      return vec3(r * cos(angle), r * sin(angle), z);
  }


  vec3 uniformlyRandomVector(float seed) {
      return uniformlyRandomDirection(seed) * sqrt(random(vec3(36.7539, 50.3658, 306.2759), seed));
}` +
  makeShadow(objects) + '\n' +
  makeCalculateColor(objects) + '\n' +
  makeMain();
  return fs;
}

////////////////////////////////////////////////////////////////////////////////
// utility functions
////////////////////////////////////////////////////////////////////////////////

function getEyeRay(matrix, x, y) {
  return matrix.multiply(Vector.create([x, y, 0, 1])).divideByW().ensure3().subtract(eye);
}

function setUniforms(program, uniforms) {
  for(var name in uniforms) {
    var value = uniforms[name];
    var location = gl.getUniformLocation(program, name);
    if(location == null) continue;
    if(value instanceof Vector) {
      gl.uniform3fv(location, new Float32Array([value.elements[0], value.elements[1], value.elements[2]]));
    } else if(value instanceof Matrix) {
      gl.uniformMatrix4fv(location, false, new Float32Array(value.flatten()));
    } else {
      gl.uniform1f(location, value);
    }
  }
}

function concat(objects, func) {
  var text = '';
  for(var i = 0; i < objects.length; i++) {
    text += func(objects[i]);
  }
  return text;
}

Vector.prototype.ensure3 = function() {
  return Vector.create([this.elements[0], this.elements[1], this.elements[2]]);
};

Vector.prototype.ensure4 = function(w) {
  return Vector.create([this.elements[0], this.elements[1], this.elements[2], w]);
};

Vector.prototype.divideByW = function() {
  var w = this.elements[this.elements.length - 1];
  var newElements = [];
  for(var i = 0; i < this.elements.length; i++) {
    newElements.push(this.elements[i] / w);
  }
  return Vector.create(newElements);
};

Vector.prototype.componentDivide = function(vector) {
  if(this.elements.length != vector.elements.length) {
    return null;
  }
  var newElements = [];
  for(var i = 0; i < this.elements.length; i++) {
    newElements.push(this.elements[i] / vector.elements[i]);
  }
  return Vector.create(newElements);
};

Vector.min = function(a, b) {
  if(a.length != b.length) {
    return null;
  }
  var newElements = [];
  for(var i = 0; i < a.elements.length; i++) {
    newElements.push(Math.min(a.elements[i], b.elements[i]));
  }
  return Vector.create(newElements);
};

Vector.max = function(a, b) {
  if(a.length != b.length) {
    return null;
  }
  var newElements = [];
  for(var i = 0; i < a.elements.length; i++) {
    newElements.push(Math.max(a.elements[i], b.elements[i]));
  }
  return Vector.create(newElements);
};

Vector.prototype.minComponent = function() {
  var value = Number.MAX_VALUE;
  for(var i = 0; i < this.elements.length; i++) {
    value = Math.min(value, this.elements[i]);
  }
  return value;
};

Vector.prototype.maxComponent = function() {
  var value = -Number.MAX_VALUE;
  for(var i = 0; i < this.elements.length; i++) {
    value = Math.max(value, this.elements[i]);
  }
  return value;
};

function compileSource(source, type) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw 'compile error: ' + gl.getShaderInfoLog(shader);
  }
  return shader;
}

function compileShader(vertexSource, fragmentSource) {
  var shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, compileSource(vertexSource, gl.VERTEX_SHADER));
  gl.attachShader(shaderProgram, compileSource(fragmentSource, gl.FRAGMENT_SHADER));
  gl.linkProgram(shaderProgram);
  if(!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    throw 'link error: ' + gl.getProgramInfoLog(shaderProgram);
  }
  return shaderProgram;
}

////////////////////////////////////////////////////////////////////////////////
// class Sphere
////////////////////////////////////////////////////////////////////////////////

function Sphere(center, radius, id) {
  this.center = center;
  this.radius = radius;
  this.centerStr = 'sphereCenter' + id;
  this.radiusStr = 'sphereRadius' + id;
  this.intersectStr = 'tSphere' + id;
  this.temporaryTranslation = Vector.create([0, 0, 0]);
}

Sphere.prototype.getGlobalCode = function() {
  return '' +
' uniform vec3 ' + this.centerStr + ';' +
' uniform float ' + this.radiusStr + ';';
};

Sphere.prototype.getIntersectCode = function() {
  return '' +
' float ' + this.intersectStr + ' = intersectSphere(origin, ray, ' + this.centerStr + ', ' + this.radiusStr + ');';
};

Sphere.prototype.getShadowTestCode = function() {
  return '' +
  this.getIntersectCode() + 
' if(' + this.intersectStr + ' < 1.0) return 0.0;';
};

Sphere.prototype.getMinimumIntersectCode = function() {
  return '' +
' if(' + this.intersectStr + ' < t) t = ' + this.intersectStr + ';';
};

Sphere.prototype.getNormalCalculationCode = function() {
  return '' +
' else if(t == ' + this.intersectStr + ') normal = normalForSphere(hit, ' + this.centerStr + ', ' + this.radiusStr + ');';
};

Sphere.prototype.setUniforms = function(renderer) {
  renderer.uniforms[this.centerStr] = this.center.add(this.temporaryTranslation);
  renderer.uniforms[this.radiusStr] = this.radius;
};

Sphere.prototype.temporaryTranslate = function(translation) {
  this.temporaryTranslation = translation;
};

Sphere.prototype.translate = function(translation) {
  this.center = this.center.add(translation);
};

Sphere.prototype.getMinCorner = function() {
  return this.center.add(this.temporaryTranslation).subtract(Vector.create([this.radius, this.radius, this.radius]));
};

Sphere.prototype.getMaxCorner = function() {
  return this.center.add(this.temporaryTranslation).add(Vector.create([this.radius, this.radius, this.radius]));
};

Sphere.prototype.intersect = function(origin, ray) {
  return Sphere.intersect(origin, ray, this.center.add(this.temporaryTranslation), this.radius);
};

Sphere.intersect = function(origin, ray, center, radius) {
  var toSphere = origin.subtract(center);
  var a = ray.dot(ray);
  var b = 2*toSphere.dot(ray);
  var c = toSphere.dot(toSphere) - radius*radius;
  var discriminant = b*b - 4*a*c;
  if(discriminant > 0) {
    var t = (-b - Math.sqrt(discriminant)) / (2*a);
    if(t > 0) {
      return t;
    }
  }
  return Number.MAX_VALUE;
};

////////////////////////////////////////////////////////////////////////////////
// class Cube
////////////////////////////////////////////////////////////////////////////////

function Cube(minCorner, maxCorner, id) {
  this.minCorner = minCorner;
  this.maxCorner = maxCorner;
  this.minStr = 'cubeMin' + id;
  this.maxStr = 'cubeMax' + id;
  this.intersectStr = 'tCube' + id;
  this.temporaryTranslation = Vector.create([0, 0, 0]);
}

Cube.prototype.getGlobalCode = function() {
  return '' +
' uniform vec3 ' + this.minStr + ';' +
' uniform vec3 ' + this.maxStr + ';';
};

Cube.prototype.getIntersectCode = function() {
  return '' +
' vec2 ' + this.intersectStr + ' = intersectCube(origin, ray, ' + this.minStr + ', ' + this.maxStr + ');';
};

Cube.prototype.getShadowTestCode = function() {
  return '' +
  this.getIntersectCode() + 
' if(' + this.intersectStr + '.x > 0.0 && ' + this.intersectStr + '.x < 1.0 && ' + this.intersectStr + '.x < ' + this.intersectStr + '.y) return 0.0;';
};

Cube.prototype.getMinimumIntersectCode = function() {
  return '' +
' if(' + this.intersectStr + '.x > 0.0 && ' + this.intersectStr + '.x < ' + this.intersectStr + '.y && ' + this.intersectStr + '.x < t) t = ' + this.intersectStr + '.x;';
};

Cube.prototype.getNormalCalculationCode = function() {
  return '' +
  // have to compare intersectStr.x < intersectStr.y otherwise two coplanar
  // cubes will look wrong (one cube will "steal" the hit from the other)
' else if(t == ' + this.intersectStr + '.x && ' + this.intersectStr + '.x < ' + this.intersectStr + '.y) normal = normalForCube(hit, ' + this.minStr + ', ' + this.maxStr + ');';
};

Cube.prototype.setUniforms = function(renderer) {
  renderer.uniforms[this.minStr] = this.getMinCorner();
  renderer.uniforms[this.maxStr] = this.getMaxCorner();
};

Cube.prototype.temporaryTranslate = function(translation) {
  this.temporaryTranslation = translation;
};

Cube.prototype.translate = function(translation) {
  this.minCorner = this.minCorner.add(translation);
  this.maxCorner = this.maxCorner.add(translation);
};

Cube.prototype.getMinCorner = function() {
  return this.minCorner.add(this.temporaryTranslation);
};

Cube.prototype.getMaxCorner = function() {
  return this.maxCorner.add(this.temporaryTranslation);
};

Cube.prototype.intersect = function(origin, ray) {
  return Cube.intersect(origin, ray, this.getMinCorner(), this.getMaxCorner());
};

Cube.intersect = function(origin, ray, cubeMin, cubeMax) {
  var tMin = cubeMin.subtract(origin).componentDivide(ray);
  var tMax = cubeMax.subtract(origin).componentDivide(ray);
  var t1 = Vector.min(tMin, tMax);
  var t2 = Vector.max(tMin, tMax);
  var tNear = t1.maxComponent();
  var tFar = t2.minComponent();
  if(tNear > 0 && tNear < tFar) {
    return tNear;
  }
  return Number.MAX_VALUE;
};

////////////////////////////////////////////////////////////////////////////////
// class Light
////////////////////////////////////////////////////////////////////////////////

function Light() {
  this.temporaryTranslation = Vector.create([0, 0, 0]);
}

Light.prototype.getGlobalCode = function() {
  return 'uniform vec3 light;';
};

Light.prototype.getIntersectCode = function() {
  return '';
};

Light.prototype.getShadowTestCode = function() {
  return '';
};

Light.prototype.getMinimumIntersectCode = function() {
  return '';
};

Light.prototype.getNormalCalculationCode = function() {
  return '';
};

Light.prototype.setUniforms = function(renderer) {
  renderer.uniforms.light = light.add(this.temporaryTranslation);
};

Light.clampPosition = function(position) {
  for(var i = 0; i < position.elements.length; i++) {
    position.elements[i] = Math.max(lightSize - 1, Math.min(1 - lightSize, position.elements[i]));
  }
};

Light.prototype.temporaryTranslate = function(translation) {
  var tempLight = light.add(translation);
  Light.clampPosition(tempLight);
  this.temporaryTranslation = tempLight.subtract(light);
};

Light.prototype.translate = function(translation) {
  light = light.add(translation);
  Light.clampPosition(light);
};

Light.prototype.getMinCorner = function() {
  return light.add(this.temporaryTranslation).subtract(Vector.create([lightSize, lightSize, lightSize]));
};

Light.prototype.getMaxCorner = function() {
  return light.add(this.temporaryTranslation).add(Vector.create([lightSize, lightSize, lightSize]));
};

Light.prototype.intersect = function(origin, ray) {
  return Number.MAX_VALUE;
};

////////////////////////////////////////////////////////////////////////////////
// class PathTracer
////////////////////////////////////////////////////////////////////////////////

function PathTracer() {
  // create vertex buffer
  var vertices = [
    -1, -1,
    -1, +1,
    +1, -1,
    +1, +1
  ];
  this.vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

  // create framebuffer
  this.pathTracerFramebuffer = gl.createFramebuffer();
  this.temporalDenoiserFramebuffer = gl.createFramebuffer();
  this.spatialDenoiserFrameBuffer = gl.createFramebuffer();

  // create textures
  var type = gl.UNSIGNED_BYTE;
  this.colorTextures = createPingPongTextures();                  // raw output of 1 spp path trace
  this.gBufferTextures = createPingPongTextures();                // vec4(normal, depth)
  this.integratedColorTextures = createPingPongTextures();        // temporally integrated color
  this.integratedColorSquareTextures = createPingPongTextures();  // temporally integrated color*color for variance estimation
  this.atrousTextures = createPingPongTextures();                 // a-trous wavelet filter iteration

  // create temporal denoiser
  this.temporalDenoiserProgram = compileShader(denoiserVertexSource, temporalDenoiserFragmentSource);
  this.renderVertexAttribute = gl.getAttribLocation(this.temporalDenoiserProgram, 'vertex');
  gl.enableVertexAttribArray(this.renderVertexAttribute);
  setTexturesLoc(this.temporalDenoiserProgram, ['colorTexture', 'gBufferTexture', 'integratedColorTexture', 'integratedColorSquareTexture', 'lastGBufferTexture']);
  
  // create spatial denoiser
  this.spatialDenoiserProgram = compileShader(denoiserVertexSource, spatialDenoiserFragmentSource);
  this.renderVertexAttribute = gl.getAttribLocation(this.spatialDenoiserProgram, 'vertex');
  gl.enableVertexAttribArray(this.renderVertexAttribute);
  setTexturesLoc(this.spatialDenoiserProgram, ['integratedColorTexture', 'gBufferTexture', 'integratedColorSquareTexture']);
  this.spatialDenoiserProgram.stepLengthLoc = gl.getUniformLocation(this.spatialDenoiserProgram, 'stepLength');

  // objects and shader will be filled in when setObjects() is called
  this.objects = [];
  this.sampleCount = 0;
  this.sampleIndex = 0;
  this.tracerProgram = null;
  
  // create 2 textures for input and output. they are swapped at each render.
  function createPingPongTextures() {
    const textures = [];
    for (let i = 0; i < 2; i++) {
        textures.push(gl.createTexture());
      gl.bindTexture(gl.TEXTURE_2D, textures[i]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 512, 512, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    return textures;
  }

  // set uniform location of textures in sequence
  function setTexturesLoc(program, textures) {
    gl.useProgram(program);
    textures.forEach( (name, i) => {
      const textureLoc = gl.getUniformLocation(program, name);
      gl.uniform1i(textureLoc, i);
    });
  }
}

PathTracer.prototype.setObjects = function(objects) {
  this.uniforms = {};
  this.sampleCount = 0;
  this.objects = objects;

  // create tracer shader
  if(this.tracerProgram != null) {
    gl.deleteProgram(this.shaderProgram);
  }
  this.tracerProgram = compileShader(tracerVertexSource, makeTracerFragmentSource(objects));
  this.tracerVertexAttribute = gl.getAttribLocation(this.tracerProgram, 'vertex');
  gl.enableVertexAttribArray(this.tracerVertexAttribute);
};


// path tracing
PathTracer.prototype.update = function(matrix, timeSinceStart) {

  gl.useProgram(this.tracerProgram);

  // calculate and set uniforms
  for(var i = 0; i < this.objects.length; i++) {
    this.objects[i].setUniforms(this);
  }
  this.uniforms.eye = eye;
  this.uniforms.glossiness = glossiness;
  this.uniforms.ray00 = getEyeRay(matrix, -1, -1);
  this.uniforms.ray01 = getEyeRay(matrix, -1, +1);
  this.uniforms.ray10 = getEyeRay(matrix, +1, -1);
  this.uniforms.ray11 = getEyeRay(matrix, +1, +1);
  this.uniforms.timeSinceStart = timeSinceStart;
  setUniforms(this.tracerProgram, this.uniforms);
  
  // bind framebuffer
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);

  // swap ping-pong textures before rendering
  this.colorTextures.reverse();
  this.gBufferTextures.reverse();
  this.integratedColorTextures.reverse();
  this.integratedColorSquareTextures.reverse();
  this.atrousTextures.reverse();

  // render to textures
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.pathTracerFramebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorTextures[0], 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.gBufferTextures[0], 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  
  // bind vertex buffer
  gl.vertexAttribPointer(this.tracerVertexAttribute, 2, gl.FLOAT, false, 0, 0);

  // render
  if (denoiseMode == 2) {   // denoiser off, just render to canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

};


// temporal and spatial denoise
PathTracer.prototype.render = function() {
  
  if (denoiseMode == 2) {   // denoiser off
    return;
  }

  //////////////////////// temporal denoiser ///////////////////////////////////
  
  gl.useProgram(this.temporalDenoiserProgram);

  // bind textures
  bindTextures([
    this.colorTextures[0],
    this.gBufferTextures[0],
    this.integratedColorTextures[0],
    this.integratedColorSquareTextures[0],
    this.gBufferTextures[1],
  ]);

  // bind framebuffer and attach textures to it for output
  
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.temporalDenoiserFramebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.integratedColorTextures[1], 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.integratedColorSquareTextures[1], 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  
  // bind vertex buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  
  // render
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  if (denoiseMode == 1) {   // denoiser off
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return;    
  }

  //////////////////////// spatial denoiser ///////////////////////////////////

  gl.useProgram(this.spatialDenoiserProgram);
  gl.uniform1i(this.spatialDenoiserProgram.stepLengthLoc, 1.0);

  // bind textures
  bindTextures([
    this.integratedColorTextures[1],
    this.gBufferTextures[0],
    this.integratedColorSquareTextures[1],
  ]);

  // bind framebuffer, vertex buffer, and render
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.spatialDenoiserFrameBuffer);
  // gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.integratedColorTextures[0], 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.atrousTextures[0], 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // /////////////////////// second pass ///////////////////////////////////////

  gl.useProgram(this.spatialDenoiserProgram);
  gl.uniform1i(this.spatialDenoiserProgram.stepLengthLoc, 2.0);

  // bind textures
  bindTextures([
    this.atrousTextures[0],
    this.gBufferTextures[0],
    this.integratedColorSquareTextures[1],
  ]);

  // bind framebuffer, vertex buffer, and render
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.spatialDenoiserFrameBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.atrousTextures[1], 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  ///////////////////////// third pass /////////////////////////////////////////

  gl.useProgram(this.spatialDenoiserProgram);
  gl.uniform1i(this.spatialDenoiserProgram.stepLengthLoc, 4.0);

  // bind textures
  bindTextures([
    // this.integratedColorTextures[1],
    this.atrousTextures[1],
    this.gBufferTextures[0],
    this.integratedColorSquareTextures[1],
  ]);

  // bind framebuffer, vertex buffer, and render
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // bind textures in sequence
  function bindTextures(textures) {
    textures.forEach((texture, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, texture);
    });
  }
};


////////////////////////////////////////////////////////////////////////////////
// class Renderer
////////////////////////////////////////////////////////////////////////////////

function Renderer() {
  var vertices = [
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    1, 1, 0,
    0, 0, 1,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1
  ];
  var indices = [
    0, 1, 1, 3, 3, 2, 2, 0,
    4, 5, 5, 7, 7, 6, 6, 4,
    0, 4, 1, 5, 2, 6, 3, 7
  ];

  // create vertex buffer
  this.vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

  // create index buffer
  this.indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

  // create line shader
  this.lineProgram = compileShader(lineVertexSource, lineFragmentSource);
  this.vertexAttribute = gl.getAttribLocation(this.lineProgram, 'vertex');
  gl.enableVertexAttribArray(this.vertexAttribute);

  this.objects = [];
  this.selectedObject = null;
  this.pathTracer = new PathTracer();
}

Renderer.prototype.setObjects = function(objects) {
  this.objects = objects;
  this.selectedObject = null;
  this.pathTracer.setObjects(objects);
};

Renderer.prototype.update = function(modelviewProjection, timeSinceStart) {
  const TAA = false;
  var jitter = Matrix.Translation(Vector.create( (TAA)? [Math.random() * 1 - 0.5, Math.random() * 1 - 0.5, 0] : [0, 0, 0]).multiply(1 / 512));
  var inverse = jitter.multiply(modelviewProjection).inverse();
  this.modelviewProjection = modelviewProjection;
  this.pathTracer.update(inverse, timeSinceStart);
};

Renderer.prototype.render = function() {
  this.pathTracer.render();

  if(this.selectedObject != null) {
    gl.useProgram(this.lineProgram);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.vertexAttribPointer(this.vertexAttribute, 3, gl.FLOAT, false, 0, 0);
    setUniforms(this.lineProgram, {
      cubeMin: this.selectedObject.getMinCorner(),
      cubeMax: this.selectedObject.getMaxCorner(),
      modelviewProjection: this.modelviewProjection
    });
    gl.drawElements(gl.LINES, 24, gl.UNSIGNED_SHORT, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////
// class UI
////////////////////////////////////////////////////////////////////////////////

function UI() {
  this.renderer = new Renderer();
  this.moving = false;
}

UI.prototype.setObjects = function(objects) {
  this.objects = objects;
  this.objects.splice(0, 0, new Light());
  this.renderer.setObjects(this.objects);
};

UI.prototype.update = function(timeSinceStart) {
  this.modelview = makeLookAt(eye.elements[0], eye.elements[1], eye.elements[2], 0, 0, 0, 0, 1, 0);
  this.projection = makePerspective(55, 1, 0.1, 100);
  this.modelviewProjection = this.projection.multiply(this.modelview);
  this.renderer.update(this.modelviewProjection, timeSinceStart);
};

UI.prototype.mouseDown = function(x, y) {
  var t;
  var origin = eye;
  var ray = getEyeRay(this.modelviewProjection.inverse(), (x / 512) * 2 - 1, 1 - (y / 512) * 2);

  // test the selection box first
  if(this.renderer.selectedObject != null) {
    var minBounds = this.renderer.selectedObject.getMinCorner();
    var maxBounds = this.renderer.selectedObject.getMaxCorner();
    t = Cube.intersect(origin, ray, minBounds, maxBounds);

    if(t < Number.MAX_VALUE) {
      var hit = origin.add(ray.multiply(t));

      if(Math.abs(hit.elements[0] - minBounds.elements[0]) < 0.001) this.movementNormal = Vector.create([-1, 0, 0]);
      else if(Math.abs(hit.elements[0] - maxBounds.elements[0]) < 0.001) this.movementNormal = Vector.create([+1, 0, 0]);
      else if(Math.abs(hit.elements[1] - minBounds.elements[1]) < 0.001) this.movementNormal = Vector.create([0, -1, 0]);
      else if(Math.abs(hit.elements[1] - maxBounds.elements[1]) < 0.001) this.movementNormal = Vector.create([0, +1, 0]);
      else if(Math.abs(hit.elements[2] - minBounds.elements[2]) < 0.001) this.movementNormal = Vector.create([0, 0, -1]);
      else this.movementNormal = Vector.create([0, 0, +1]);

      this.movementDistance = this.movementNormal.dot(hit);
      this.originalHit = hit;
      this.moving = true;

      return true;
    }
  }

  t = Number.MAX_VALUE;
  this.renderer.selectedObject = null;

  for(var i = 0; i < this.objects.length; i++) {
    var objectT = this.objects[i].intersect(origin, ray);
    if(objectT < t) {
      t = objectT;
      this.renderer.selectedObject = this.objects[i];
    }
  }

  return (t < Number.MAX_VALUE);
};

UI.prototype.mouseMove = function(x, y) {
  if(this.moving) {
    var origin = eye;
    var ray = getEyeRay(this.modelviewProjection.inverse(), (x / 512) * 2 - 1, 1 - (y / 512) * 2);

    var t = (this.movementDistance - this.movementNormal.dot(origin)) / this.movementNormal.dot(ray);
    var hit = origin.add(ray.multiply(t));
    this.renderer.selectedObject.temporaryTranslate(hit.subtract(this.originalHit));

    // clear the sample buffer
    this.renderer.pathTracer.sampleCount = 0;
  }
};

UI.prototype.mouseUp = function(x, y) {
  if(this.moving) {
    var origin = eye;
    var ray = getEyeRay(this.modelviewProjection.inverse(), (x / 512) * 2 - 1, 1 - (y / 512) * 2);

    var t = (this.movementDistance - this.movementNormal.dot(origin)) / this.movementNormal.dot(ray);
    var hit = origin.add(ray.multiply(t));
    this.renderer.selectedObject.temporaryTranslate(Vector.create([0, 0, 0]));
    this.renderer.selectedObject.translate(hit.subtract(this.originalHit));
    this.moving = false;
  }
};

UI.prototype.render = function() {
  this.renderer.render();
};

UI.prototype.selectLight = function() {
  this.renderer.selectedObject = this.objects[0];
};

UI.prototype.addSphere = function() {
  this.objects.push(new Sphere(Vector.create([0, 0, 0]), 0.25, nextObjectId++));
  this.renderer.setObjects(this.objects);
};

UI.prototype.addCube = function() {
  this.objects.push(new Cube(Vector.create([-0.25, -0.25, -0.25]), Vector.create([0.25, 0.25, 0.25]), nextObjectId++));
  this.renderer.setObjects(this.objects);
};

UI.prototype.deleteSelection = function() {
  for(var i = 0; i < this.objects.length; i++) {
    if(this.renderer.selectedObject == this.objects[i]) {
      this.objects.splice(i, 1);
      this.renderer.selectedObject = null;
      this.renderer.setObjects(this.objects);
      break;
    }
  }
};

UI.prototype.updateMaterial = function() {
  var newMaterial = parseInt(document.getElementById('material').value, 10);
  if(material != newMaterial) {
    material = newMaterial;
    this.renderer.setObjects(this.objects);
  }
};

UI.prototype.updateEnvironment = function() {
  var newEnvironment = parseInt(document.getElementById('environment').value, 10);
  if(environment != newEnvironment) {
    environment = newEnvironment;
    this.renderer.setObjects(this.objects);
  }
};

UI.prototype.updateGlossiness = function() {
  var newGlossiness = parseFloat(document.getElementById('glossiness').value);
  if(isNaN(newGlossiness)) newGlossiness = 0;
  newGlossiness = Math.max(0, Math.min(1, newGlossiness));
  if(material == MATERIAL_GLOSSY && glossiness != newGlossiness) {
    this.renderer.pathTracer.sampleCount = 0;
  }
  glossiness = newGlossiness;
};

UI.prototype.updateDenoiseMode = function() {
  var newDenoiseMode = parseInt(document.getElementById('denoise').value, 10);
  if(denoiseMode != newDenoiseMode) {
    denoiseMode = newDenoiseMode;
  }
};

////////////////////////////////////////////////////////////////////////////////
// main program
////////////////////////////////////////////////////////////////////////////////

var gl;
var ui;
var error;
var canvas;
var inputFocusCount = 0;

var angleX = 0;
var angleY = 0;
var zoomZ = 2.5;
var eye = Vector.create([0, 0, 0]);
var light = Vector.create([0.4, 0.5, -0.6]);

var nextObjectId = 0;

var MATERIAL_DIFFUSE = 0;
var MATERIAL_MIRROR = 1;
var MATERIAL_GLOSSY = 2;
var material = MATERIAL_DIFFUSE;
var glossiness = 0.6;
var denoiseMode = 0

var YELLOW_BLUE_CORNELL_BOX = 0;
var RED_GREEN_CORNELL_BOX = 1;
var environment = YELLOW_BLUE_CORNELL_BOX;

function tick(timeSinceStart) {
  eye.elements[0] = zoomZ * Math.sin(angleY) * Math.cos(angleX);
  eye.elements[1] = zoomZ * Math.sin(angleX);
  eye.elements[2] = zoomZ * Math.cos(angleY) * Math.cos(angleX);

  document.getElementById('glossiness-factor').style.display = (material == MATERIAL_GLOSSY) ? 'inline' : 'none';

  ui.updateMaterial();
  ui.updateGlossiness();
  ui.updateEnvironment();
  ui.updateDenoiseMode();
  ui.update(timeSinceStart);
  ui.render();
}

function makeStacks() {
  var objects = [];

  // lower level
  objects.push(new Cube(Vector.create([-0.5, -0.75, -0.5]), Vector.create([0.5, -0.7, 0.5]), nextObjectId++));

  // further poles
  objects.push(new Cube(Vector.create([-0.45, -1, -0.45]), Vector.create([-0.4, -0.45, -0.4]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.4, -1, -0.45]), Vector.create([0.45, -0.45, -0.4]), nextObjectId++));
  objects.push(new Cube(Vector.create([-0.45, -1, 0.4]), Vector.create([-0.4, -0.45, 0.45]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.4, -1, 0.4]), Vector.create([0.45, -0.45, 0.45]), nextObjectId++));

  // upper level
  objects.push(new Cube(Vector.create([-0.3, -0.5, -0.3]), Vector.create([0.3, -0.45, 0.3]), nextObjectId++));

  // closer poles
  objects.push(new Cube(Vector.create([-0.25, -0.7, -0.25]), Vector.create([-0.2, -0.25, -0.2]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.2, -0.7, -0.25]), Vector.create([0.25, -0.25, -0.2]), nextObjectId++));
  objects.push(new Cube(Vector.create([-0.25, -0.7, 0.2]), Vector.create([-0.2, -0.25, 0.25]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.2, -0.7, 0.2]), Vector.create([0.25, -0.25, 0.25]), nextObjectId++));

  // upper level
  objects.push(new Cube(Vector.create([-0.25, -0.25, -0.25]), Vector.create([0.25, -0.2, 0.25]), nextObjectId++));

  return objects;
}

function makeTableAndChair() {
  var objects = [];

  // table top
  objects.push(new Cube(Vector.create([-0.5, -0.35, -0.5]), Vector.create([0.3, -0.3, 0.5]), nextObjectId++));

  // table legs
  objects.push(new Cube(Vector.create([-0.45, -1, -0.45]), Vector.create([-0.4, -0.35, -0.4]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.2, -1, -0.45]), Vector.create([0.25, -0.35, -0.4]), nextObjectId++));
  objects.push(new Cube(Vector.create([-0.45, -1, 0.4]), Vector.create([-0.4, -0.35, 0.45]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.2, -1, 0.4]), Vector.create([0.25, -0.35, 0.45]), nextObjectId++));

  // chair seat
  objects.push(new Cube(Vector.create([0.3, -0.6, -0.2]), Vector.create([0.7, -0.55, 0.2]), nextObjectId++));

  // chair legs
  objects.push(new Cube(Vector.create([0.3, -1, -0.2]), Vector.create([0.35, -0.6, -0.15]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.3, -1, 0.15]), Vector.create([0.35, -0.6, 0.2]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.65, -1, -0.2]), Vector.create([0.7, 0.1, -0.15]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.65, -1, 0.15]), Vector.create([0.7, 0.1, 0.2]), nextObjectId++));

  // chair back
  objects.push(new Cube(Vector.create([0.65, 0.05, -0.15]), Vector.create([0.7, 0.1, 0.15]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.65, -0.55, -0.09]), Vector.create([0.7, 0.1, -0.03]), nextObjectId++));
  objects.push(new Cube(Vector.create([0.65, -0.55, 0.03]), Vector.create([0.7, 0.1, 0.09]), nextObjectId++));

  // sphere on table
  objects.push(new Sphere(Vector.create([-0.1, -0.05, 0]), 0.25, nextObjectId++));

  return objects;
}

function makeSphereAndCube() {
  var objects = [];
  objects.push(new Cube(Vector.create([-0.25, -1, -0.25]), Vector.create([0.25, -0.75, 0.25]), nextObjectId++));
  objects.push(new Sphere(Vector.create([0, -0.75, 0]), 0.25, nextObjectId++));
  return objects;
}

function makeSphereColumn() {
  var objects = [];
  objects.push(new Sphere(Vector.create([0, 0.75, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, 0.25, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, -0.25, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, -0.75, 0]), 0.25, nextObjectId++));
  return objects;
}

function makeCubeAndSpheres() {
  var objects = [];
  objects.push(new Cube(Vector.create([-0.25, -0.25, -0.25]), Vector.create([0.25, 0.25, 0.25]), nextObjectId++));
  objects.push(new Sphere(Vector.create([-0.25, 0, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([+0.25, 0, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, -0.25, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, +0.25, 0]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, 0, -0.25]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0, 0, +0.25]), 0.25, nextObjectId++));
  return objects;
}

function makeSpherePyramid() {
  var root3_over4 = 0.433012701892219;
  var root3_over6 = 0.288675134594813;
  var root6_over6 = 0.408248290463863;
  var objects = [];

  // first level
  objects.push(new Sphere(Vector.create([-0.5, -0.75, -root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0.0, -0.75, -root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0.5, -0.75, -root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([-0.25, -0.75, root3_over4 - root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0.25, -0.75, root3_over4 - root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0.0, -0.75, 2.0 * root3_over4 - root3_over6]), 0.25, nextObjectId++));

  // second level
  objects.push(new Sphere(Vector.create([0.0, -0.75 + root6_over6, root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([-0.25, -0.75 + root6_over6, -0.5 * root3_over6]), 0.25, nextObjectId++));
  objects.push(new Sphere(Vector.create([0.25, -0.75 + root6_over6, -0.5 * root3_over6]), 0.25, nextObjectId++));

  // third level
  objects.push(new Sphere(Vector.create([0.0, -0.75 + 2.0 * root6_over6, 0.0]), 0.25, nextObjectId++));

  return objects;
}

var XNEG = 0, XPOS = 1, YNEG = 2, YPOS = 3, ZNEG = 4, ZPOS = 5;

function addRecursiveSpheresBranch(objects, center, radius, depth, dir) {
  objects.push(new Sphere(center, radius, nextObjectId++));
  if(depth--) {
    if(dir != XNEG) addRecursiveSpheresBranch(objects, center.subtract(Vector.create([radius * 1.5, 0, 0])), radius / 2, depth, XPOS);
    if(dir != XPOS) addRecursiveSpheresBranch(objects, center.add(Vector.create([radius * 1.5, 0, 0])),      radius / 2, depth, XNEG);
    
    if(dir != YNEG) addRecursiveSpheresBranch(objects, center.subtract(Vector.create([0, radius * 1.5, 0])), radius / 2, depth, YPOS);
    if(dir != YPOS) addRecursiveSpheresBranch(objects, center.add(Vector.create([0, radius * 1.5, 0])),      radius / 2, depth, YNEG);
    
    if(dir != ZNEG) addRecursiveSpheresBranch(objects, center.subtract(Vector.create([0, 0, radius * 1.5])), radius / 2, depth, ZPOS);
    if(dir != ZPOS) addRecursiveSpheresBranch(objects, center.add(Vector.create([0, 0, radius * 1.5])),      radius / 2, depth, ZNEG);
  }
}

function makeRecursiveSpheres() {
  var objects = [];
  addRecursiveSpheresBranch(objects, Vector.create([0, 0, 0]), 0.3, 2, -1);
  return objects;
}

window.onload = function() {
  gl = null;
  error = document.getElementById('error');
  canvas = document.getElementById('canvas');
  try { 
    gl = canvas.getContext('webgl2');
  } catch(e) {
    console.error(e);
  }

  if(gl) {
    error.innerHTML = 'Loading...';

    // keep track of whether an <input> is focused or not (will be no only if inputFocusCount == 0)
    var inputs = document.getElementsByTagName('input');
    for(var i= 0; i < inputs.length; i++) {
      inputs[i].onfocus = function(){ inputFocusCount++; };
      inputs[i].onblur = function(){ inputFocusCount--; };
    }

    material = parseInt(document.getElementById('material').value, 10);
    environment = parseInt(document.getElementById('environment').value, 10);
    ui = new UI();
    ui.setObjects(makeSphereColumn());
    var start = new Date();
    error.style.zIndex = -1;

    var stats = new Stats();
    stats.setMode(1);
    stats.domElement.style.left = '0px';
    stats.domElement.style.top = '0px';
    document.body.appendChild(stats.domElement);
    

    setInterval(function(){ 
      stats.begin(); 
      tick((new Date() - start) * 0.001); 
      stats.end();
    }, 1000 / 60);
  } else {
    error.innerHTML = 'Your browser does not support WebGL.<br>Please see <a href="http://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation">Getting a WebGL Implementation</a>.';
  }
};

function elementPos(element) {
  var x = 0, y = 0;
  while(element.offsetParent) {
    x += element.offsetLeft;
    y += element.offsetTop;
    element = element.offsetParent;
  }
  return { x: x, y: y };
}

function eventPos(event) {
  return {
    x: event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft,
    y: event.clientY + document.body.scrollTop + document.documentElement.scrollTop
  };
}

function canvasMousePos(event) {
  var mousePos = eventPos(event);
  var canvasPos = elementPos(canvas);
  return {
    x: mousePos.x - canvasPos.x,
    y: mousePos.y - canvasPos.y
  };
}

var mouseDown = false, oldX, oldY;

document.onmousedown = function(event) {
  var mouse = canvasMousePos(event);
  oldX = mouse.x;
  oldY = mouse.y;

  if(mouse.x >= 0 && mouse.x < 512 && mouse.y >= 0 && mouse.y < 512) {
    mouseDown = !ui.mouseDown(mouse.x, mouse.y);

    // disable selection because dragging is used for rotating the camera and moving objects
    return false;
  }

  return true;
};

document.onmousemove = function(event) {
  var mouse = canvasMousePos(event);

  if(mouseDown) {
    // update the angles based on how far we moved since last time
    angleY -= (mouse.x - oldX) * 0.01;
    angleX += (mouse.y - oldY) * 0.01;

    // don't go upside down
    angleX = Math.max(angleX, -Math.PI / 2 + 0.01);
    angleX = Math.min(angleX, Math.PI / 2 - 0.01);

    // clear the sample buffer
    ui.renderer.pathTracer.sampleCount = 0;

    // remember this coordinate
    oldX = mouse.x;
    oldY = mouse.y;
  } else {
    var canvasPos = elementPos(canvas);
    ui.mouseMove(mouse.x, mouse.y);
  }
};

document.onmouseup = function(event) {
  mouseDown = false;

  var mouse = canvasMousePos(event);
  ui.mouseUp(mouse.x, mouse.y);
};

document.onkeydown = function(event) {
  // if there are no <input> elements focused
  if(inputFocusCount == 0) {
    // if backspace or delete was pressed
    if(event.keyCode == 8 || event.keyCode == 46) {
      ui.deleteSelection();

      // don't let the backspace key go back a page
      return false;
    }
  }
};