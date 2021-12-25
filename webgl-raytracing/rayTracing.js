/*
 WebGL Path Tracing (http://madebyevan.com/webgl-path-tracing/)
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

// vertex shader for drawing a textured quad
var renderVertexSource = 
`#version 300 es
in vec3 vertex; 
out vec2 texCoord; 
void main() {
    texCoord = vertex.xy * 0.5 + 0.5;
    gl_Position = vec4(vertex, 1.0);
}`;

// fragment shader for drawing a textured quad
var renderFragmentSource =
`#version 300 es
precision highp float;
out vec4 fragColor;
in vec2 texCoord; 
uniform sampler2D colorTexture;
uniform sampler2D rawColorTexture;
void main() {
    vec4 color = texture(colorTexture, texCoord);
    float variance = color.w;
    color.w = 1.0;
    vec4 rawColor = texture(rawColorTexture, texCoord);
    
    fragColor = color;


    float kernal[25] = float[](0.01247764154323288, 0.02641516735431067, 0.03391774626899505, 0.02641516735431067, 0.01247764154323288, 
      0.02641516735431067, 0.05592090972790157, 0.07180386941492609, 0.05592090972790157, 0.02641516735431067, 
      0.03391774626899505, 0.07180386941492609, 0.09219799334529226, 0.07180386941492609, 0.03391774626899505, 
      0.02641516735431067, 0.05592090972790157, 0.07180386941492609, 0.05592090972790157, 0.02641516735431067, 
      0.01247764154323288, 0.02641516735431067, 0.03391774626899505, 0.02641516735431067, 0.01247764154323288);
    vec4 gaussianColor = vec4(0,0,0,0);
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        int index = i * 5 + j;
        gaussianColor += kernal[index] * texture(colorTexture, texCoord + vec2((1.0 / 512.0 * float(i - 2)), (1.0 / 512.0 * float(j - 2))));
      }
    }
    variance = gaussianColor.w;

    fragColor = mix(color, gaussianColor, variance);
    fragColor.w = 1.0;
    // if (texCoord.x < 0.5) 
      // fragColor = vec4(variance, variance, variance, 1.0);
      // fragColor = rawColor;
      fragColor = color;
}`;

// fragment shader for drawing a textured quad
var denoiserFragmentSource =
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
void main() {
    vec4 rawColor = texture(colorTexture, texCoord);
    vec4 rawColorSquare = rawColor * rawColor;
    vec4 gBuffer = texture(gBufferTexture, texCoord);
    vec4 integratedColor = texture(integratedColorTexture, texCoord);
    vec4 integratedColorSquare = texture(integratedColorSquareTexture, texCoord);
    vec4 lastGBuffer = texture(lastGBufferTexture, texCoord);

    vec4 varianceVec = vec4((integratedColorSquare - integratedColor * integratedColor).xyz, 0.0);
    float variance = dot(varianceVec, varianceVec);

    float deltaDepth = abs(lastGBuffer.w - gBuffer.w);
    
    vec3 deltaNormal = gBuffer.xyz - lastGBuffer.xyz;
    float dotDeltaNormal = dot(deltaNormal, deltaNormal);

    vec4 clampColorMin = vec4(1,1,1,1);
    vec4 clampColorMax = vec4(0,0,0,1);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        // avgColor += (1.0 / 9.0) * texture(colorTexture, texCoord + vec2((1.0 / 512.0 * float(i - 1)), (1.0 / 512.0 * float(j - 1))));
        vec4 neighbor = texture(colorTexture, texCoord + vec2((1.0 / 512.0 * float(i - 1)), (1.0 / 512.0 * float(j - 1))));
        clampColorMin = vec4(min(clampColorMin.x, neighbor.x), min(clampColorMin.y, neighbor.y), min(clampColorMin.z, neighbor.z), 1.0);
        clampColorMax = vec4(max(clampColorMax.x, neighbor.x), max(clampColorMax.y, neighbor.y), max(clampColorMax.z, neighbor.z), 1.0);
      }
    }
    vec4 clampThreshold = vec4(0.05, 0.05, 0.05, 1.0);
    clampColorMin -= clampThreshold;
    clampColorMax += clampThreshold;

    float mixWeight = 0.0;
    if (lessThanEqual(clampColorMin, integratedColor) == bvec4(true,true,true,true) && greaterThanEqual(clampColorMax, integratedColor) == bvec4(true,true,true,true)) {
      mixWeight = 0.9;
    } else if (deltaDepth < 0.01 && dotDeltaNormal < 0.01) {
      // same geometry, only light changes
      mixWeight = 0.2;
      variance = 1.0;
    } else {
      // disocclusion
      mixWeight = 0.0;
      variance = 0.0;
    }
    fragColor = mix(rawColor, integratedColor, mixWeight) + vec4(0,0,0,1);
    colorSquare = mix(rawColorSquare, integratedColorSquare, mixWeight) + vec4(0,0,0,1);
    fragColor.w = variance * 100.0;


    // fragColor = rawColor;
    // fragColor = avgColor;
    // fragColor = gaussianColor;
    // fragColor = vec4((vec3(1.0, 1.0, 1.0) * gBuffer.w), 1.0); // depth
    // fragColor = vec4(gBuffer.xyz, 1.0);  // normal
    // fragColor = vec4(variance, variance, variance, 0.0) * 1.0 + vec4(0,0,0,1);
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
    // vec4 lastColor = texture(colorTexture, texCoord).rgba;
    // vec4 lastGBuffer = texture(gBufferTexture, texCoord).rgba;
    // vec4 lastColorSquare = texture(colorSquareTexture, texCoord).rgba;
    vec3 color; vec3 normal; float depth;
    calculateColor(eye, initialRay, newLight, color, normal, depth);

    // float deltaDepth = abs(lastGBuffer.w - depth);
    // vec3 deltaNormal = normal - lastGBuffer.xyz;
    // float dotDeltaNormal = dot(deltaNormal, deltaNormal);

    // vec4 avgColor = vec4(0,0,0,1);
    // for (int i = 0; i < 3; ++i) {
    //   for (int j = 0; j < 3; ++j) {
    //     avgColor += (1.0 / 9.0) * texture(colorTexture, texCoord + vec2((1.0 / 512.0 * float(i - 1)), (1.0 / 512.0 * float(j - 1))));
    //   }
    // }
    // vec4 deltaColor = avgColor - lastColor;
    // float dotDeltaColor = dot(deltaColor, deltaColor);

    // if (deltaDepth < 0.01 && dotDeltaNormal < 0.1) {
    //   //fragColor = vec4(1, 1, 1, 1.0);
    //   fragColor = vec4(mix(color, lastColor.xyz, 0.8), 1.0);
    //   colorSquare = vec4(mix(color * color, lastColorSquare.xyz, 0.8), 1.0);
    // } else {
      //fragColor = vec4(dotDeltaNormal,dotDeltaNormal,dotDeltaNormal, 1.0);
      //fragColor = vec4(0,0,0, 1.0);
      fragColor = vec4(color, 1.0);
      colorSquare = vec4(color * color, 1.0);
    // }
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
  // uniform sampler2D colorTexture;
  // uniform sampler2D gBufferTexture; 
  // uniform sampler2D colorSquareTexture;
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
  this.framebuffer = gl.createFramebuffer();

  // create textures
  var type = gl.UNSIGNED_BYTE;
  this.colorTextures = [];
  for (var i = 0; i < 2; i++) {
      this.colorTextures.push(gl.createTexture());
    gl.bindTexture(gl.TEXTURE_2D, this.colorTextures[i]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 512, 512, 0, gl.RGBA, type, null);
  }

  // this.colorSquareTextures = [];
  // for (var i = 0; i < 2; i++) {
  //     this.colorSquareTextures.push(gl.createTexture());
  //   gl.bindTexture(gl.TEXTURE_2D, this.colorSquareTextures[i]);
  //   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  //   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  //   gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 512, 512, 0, gl.RGBA, type, null);
  // }

  this.gBufferTextures = [];
  for (var i = 0; i < 2; i++) {
    this.gBufferTextures.push(gl.createTexture());
    gl.bindTexture(gl.TEXTURE_2D, this.gBufferTextures[i]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 512, 512, 0, gl.RGBA, type, null);
  }

  this.integratedColorTextures = [];
  for (var i = 0; i < 2; i++) {
    this.integratedColorTextures.push(gl.createTexture());
    gl.bindTexture(gl.TEXTURE_2D, this.integratedColorTextures[i]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 512, 512, 0, gl.RGBA, type, null);
  }

  this.integratedColorSquareTextures = [];
  for (var i = 0; i < 2; i++) {
    this.integratedColorSquareTextures.push(gl.createTexture());
    gl.bindTexture(gl.TEXTURE_2D, this.integratedColorSquareTextures[i]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 512, 512, 0, gl.RGBA, type, null);
  }

  gl.bindTexture(gl.TEXTURE_2D, null);

  // create render shader
  this.renderProgram = compileShader(renderVertexSource, renderFragmentSource);
  this.renderVertexAttribute = gl.getAttribLocation(this.renderProgram, 'vertex');
  gl.enableVertexAttribArray(this.renderVertexAttribute);
  gl.useProgram(this.renderProgram);
  const colorTextureLoc = gl.getUniformLocation(this.renderProgram, "colorTexture");
  gl.uniform1i(colorTextureLoc, 0);
  const gBufferTextureLoc = gl.getUniformLocation(this.renderProgram, "rawColorTexture");
  gl.uniform1i(gBufferTextureLoc, 1);

  // create denoiser shader
  this.denoiserProgram = compileShader(renderVertexSource, denoiserFragmentSource);
  this.renderVertexAttribute = gl.getAttribLocation(this.denoiserProgram, 'vertex');
  gl.enableVertexAttribArray(this.renderVertexAttribute);
  gl.useProgram(this.denoiserProgram);
  const denoiserColorTextureLoc = gl.getUniformLocation(this.denoiserProgram, "colorTexture");
  gl.uniform1i(denoiserColorTextureLoc, 0);
  const denoiserGBufferTextureLoc = gl.getUniformLocation(this.denoiserProgram, "gBufferTexture");
  gl.uniform1i(denoiserGBufferTextureLoc, 1);
  const denoiserIntegratedColorTextureLoc = gl.getUniformLocation(this.denoiserProgram, "integratedColorTexture");
  gl.uniform1i(denoiserIntegratedColorTextureLoc, 2);
  const denoiserIntegratedColorSquareTextureLoc = gl.getUniformLocation(this.denoiserProgram, "integratedColorSquareTexture");
  gl.uniform1i(denoiserIntegratedColorSquareTextureLoc, 3);
  const denoiserLastGBufferTextureLoc = gl.getUniformLocation(this.denoiserProgram, "lastGBufferTexture");
  gl.uniform1i(denoiserLastGBufferTextureLoc, 4);


  // objects and shader will be filled in when setObjects() is called
  this.objects = [];
  this.sampleCount = 0;
  this.sampleIndex = 0;
  this.tracerProgram = null;
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
  
  // gl.useProgram(this.tracerProgram);
  // const colorTextureLoc = gl.getUniformLocation(this.tracerProgram, "colorTexture");
  // gl.uniform1i(colorTextureLoc, 0);
  // const gBufferTextureLoc = gl.getUniformLocation(this.tracerProgram, "gBufferTexture");
  // gl.uniform1i(gBufferTextureLoc, 1);
  // const colorSquareTextureLoc = gl.getUniformLocation(this.tracerProgram, "colorSquareTexture");
  // gl.uniform1i(colorSquareTextureLoc, 2);
};

PathTracer.prototype.update = function(matrix, timeSinceStart) {
  // calculate uniforms
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

  // set uniforms
  gl.useProgram(this.tracerProgram);
  setUniforms(this.tracerProgram, this.uniforms);
  
  // // bind texture
  // gl.activeTexture(gl.TEXTURE0);
  // gl.bindTexture(gl.TEXTURE_2D, this.colorTextures[0]);
  
  // gl.activeTexture(gl.TEXTURE1);
  // gl.bindTexture(gl.TEXTURE_2D, this.gBufferTextures[0]);

  // gl.activeTexture(gl.TEXTURE2);
  // gl.bindTexture(gl.TEXTURE_2D, this.colorSquareTextures[0]);
  
  // render to texture
  this.colorTextures.reverse();
  this.gBufferTextures.reverse();
  // this.colorSquareTextures.reverse();
  this.integratedColorTextures.reverse();
  this.integratedColorSquareTextures.reverse();

  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);

  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorTextures[0], 0);  // path tracing result
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.gBufferTextures[0], 0);  // normal and depth
  // gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, this.colorSquareTextures[0], 0);  // color^2
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  // gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);

  gl.vertexAttribPointer(this.tracerVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
};

PathTracer.prototype.render = function() {
  // denoise program
  gl.useProgram(this.denoiserProgram);

  // bind texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, this.colorTextures[0]);
  
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, this.gBufferTextures[0]);
  
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, this.integratedColorTextures[0]);
  
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, this.integratedColorSquareTextures[0]);

  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_2D, this.gBufferTextures[1]);

  // render to texture
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.integratedColorTextures[1], 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.integratedColorSquareTextures[1], 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // render program
  gl.useProgram(this.renderProgram);

  // bind texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, this.integratedColorTextures[1]);
  
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, this.colorTextures[0]);

  // gl.activeTexture(gl.TEXTURE2);
  // gl.bindTexture(gl.TEXTURE_2D, this.colorSquareTextures[0]);

  // gl.activeTexture(gl.TEXTURE2);
  // gl.bindTexture(gl.TEXTURE_2D, this.colorTextures[1]);

  // gl.activeTexture(gl.TEXTURE3);
  // gl.bindTexture(gl.TEXTURE_2D, this.gBufferTextures[1]);

  // gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
  
  gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
  gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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
  var jitter = Matrix.Translation(Vector.create([Math.random() * 2 - 1, Math.random() * 2 - 1, 0]).multiply(1 / 512));
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