// homomorphic
function homoProcessInputImg(image) {
    let [M, N] = image.shape;
    let result = nj.zeros([M, N]);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let val = Math.log(image.get(i, j) / 255 + 0.01);
            result.set(i, j, val);
        } 
    }
    return result;
}
function homoProcessOutputImg(image) {
    let [M, N] = image.shape;
    let result = nj.zeros([M, N]);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let val = 255 * (Math.E**(image.get(i, j)) - 0.01);
            result.set(i, j, val);
        }
    }
    return result;
}
function funclaplacianEnhanceImage(originalImage, lap, c=-1) {
    let [M, N] = originalImage.shape;
    let imageResult = nj.zeros([M, N]);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let value = originalImage.get(i, j) + c * (lap.get(i, j) / 255);
            if (value < 0) {
                value = 0;
            }
            else if (value > 255) {
                value = 255;
            }
            imageResult.set(i, j, value)
        }
    }
    return imageResult;
}
function laplacianImage(lap) {
    let min = lap.min();
    let max = lap.max();
    let OldRange = max - min;
    let NewRange = 255 - 0;
    let lapScaled = nj.multiply(nj.divide(nj.subtract(lap, min), OldRange), NewRange).add(0);
    return lapScaled;
}
function showLaplacianFilter(H) {
    let [P, Q] = H.shape;
    let H_normalized = nj.zeros([P, Q]);
    let min = H.min();
    let max = H.max();
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H_normalized.set(u, v, 255 * (H.get(u, v) - min) / (max - min));
        }
    }
    return H_normalized;
}
function padded_image(image) {
    let [M, N] = image.shape;
    let [P, Q] = [2 * M, 2 * N];
    let paddedImage = nj.zeros([P, Q]);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            paddedImage.set(i, j, image.get(i, j));
        }
    }
    return paddedImage;
}
function increase_brightness(image, value=35) {
    let [M, N] = image.shape;
    let imgResult = nj.zeros([M, N]);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            imgResult.set(i, j, image.get(i, j) + value);
        }
    }
    return imgResult;
}
function Fourier_transform(padded_image, H) {
    let img = padded_image.tolist().map(row => row.map(value => [value, 0]));
    let fft = nj.fft(img);
    let G = nj.multiply(H.tolist().map(row => row.map(value => [value, 0])), fft);
    let ifft = nj.ifft(G);
    let real_ifft = nj.array(ifft.tolist().map(row => row.map(value => Math.round(value[0]))));
    return real_ifft;
}
// áp dụng bộ lọc
function apply_filter(padded_image, H) {
    let [P, Q] = padded_image.shape;
    for (let x = 0; x < P; x++) {
        for (let y = 0; y < Q; y++) {
            padded_image.set(x, y, padded_image.get(x, y) * Math.pow(-1, x + y));
        }
    }
    let G = Fourier_transform(padded_image, H);
    for (let x = 0; x < P; x++) {
        for (let y = 0; y < Q; y++) {
            G.set(x, y, G.get(x, y) * Math.pow(-1, x + y));
        }
    }

    let result_image = nj.zeros([P/2, Q/2]);
    for (let i = 0; i < P/2; i++) {
        for (let j = 0; j < Q/2; j++) {
            result_image.set(i, j, G.get(i, j));
        }
    }
    return result_image;
}

function show_filter(H) {
    let [P, Q] = H.shape;
    let H1 = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H1.set(u, v, H.get(u, v) * 255);
        }
    }
    return H1;
}
// low pass
function ilpf(padded_image, D0) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            let D = Math.sqrt(Math.pow(u - (P / 2), 2) + Math.pow(v - (Q / 2), 2));
            if (D <= D0) {
                H.set(u, v, 1);
            }
        }
    }
    return H;
}
function blpf(padded_image, D0, n) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 / (1 + (((u - P / 2)**2 + (v - Q / 2)**2) / D0**2)**n));
        }
    }
    return H;
}
function glpf(padded_image, D0) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, Math.E**(-((u - P / 2)**2 + (v - Q / 2)**2) / (2 * D0**2)));
        }
    }
    return H;
}
// high pass
function ihpf(padded_image, D0) {
    let [P, Q] = padded_image.shape;
    let H = ilpf(padded_image, D0);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 - H.get(u, v));
        }
    }
    return H;
}
function bhpf(padded_image, D0, n) {
    let [P, Q] = padded_image.shape;
    let H = blpf(padded_image, D0, n);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 - H.get(u, v));
        }
    }
    return H;
}
function ghpf(padded_image, D0) {
    let [P, Q] = padded_image.shape;
    let H = glpf(padded_image, D0);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 - H.get(u, v));
        }
    }
    return H;
}
// band pass
function ibpf(padded_image, D0, D1) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            let D = Math.sqrt(Math.pow(u - (P / 2), 2) + Math.pow(v - (Q / 2), 2));
            if (D >= D0 && D <= D1) {
                H.set(u, v, 1);
            }
        }
    }
    return H;
}
function bbpf(padded_image, Dl, Dh, n) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    let W = Dh - Dl;
    let D0 = (Dl + Dh) / 2;
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            let D = Math.sqrt(Math.pow(u - (P / 2), 2) + Math.pow(v - (Q / 2), 2));
            let value = 1 - 1 / (1 + Math.pow((D * W) / (Math.pow(D, 2) - Math.pow(D0, 2)), 2 * n));
            H.set(u, v, value);
        }
    }
    return H;
}
function gbpf(padded_image, Dl, Dh) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    let W = Dh - Dl;
    let D0 = (Dl + Dh) / 2;
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            let D = Math.sqrt(Math.pow(u - (P / 2), 2) + Math.pow(v - (Q / 2), 2));
            let value = Math.E**(-Math.pow((D**2 - D0**2) / (D * W), 2));
            H.set(u, v, value);
        }
    }
    return H;
}
// band stop
function ibsf(padded_image, D0, D1) {   //ideal bandstop filter
    let [P, Q] = padded_image.shape;
    let H = ibpf(padded_image, D0, D1);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 - H.get(u, v));
        }
    }
    return H;
}
function bbsf(padded_image, D0, D1, n) {   //butterworth bandstop filter
    let [P, Q] = padded_image.shape;
    let H = bbpf(padded_image, D0, D1, n);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 - H.get(u, v));
        }
    }
    return H;
}
function gbsf(padded_image, D0, D1) {       //gaussian bandstop filter
    let [P, Q] = padded_image.shape;
    let H = gbpf(padded_image, D0, D1);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            H.set(u, v, 1 - H.get(u, v));
        }
    }
    return H;
}
// laplacian
function laplacian(padded_image) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            let D = Math.sqrt(Math.pow(u - (P / 2), 2) + Math.pow(v - (Q / 2), 2));
            H.set(u, v, -4 * Math.PI * Math.PI * D * D);
        }
    }
    return H;
}
// homomorphic
function homomorphic(padded_image, y_low=0.5, y_high=2.5, D0=32, c=1) {
    let [P, Q] = padded_image.shape;
    let H = nj.zeros([P, Q]);
    for (let u = 0; u < P; u++) {
        for (let v = 0; v < Q; v++) {
            let value = (y_high - y_low) * (1 - Math.E**(-c * ((u - P / 2)**2 + (v - Q / 2)**2) / (2 * D0**2))) + y_low
            H.set(u, v, value);
        }
    }
    return H;
}
// chỉnh kích thước ảnh cho phù hợp
function resizeImage(image, maxWidth=500, maxHeight=500) {
    let width = image.width;
    let height = image.height;

    if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height);
        width = Math.round(width * ratio);
        height = Math.round(height * ratio);
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, width, height);

    return canvas;
}
function isGrayScaleImage(mat) {
    let width = mat.cols;
    let height = mat.rows;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let pixel = mat.ucharPtr(y, x);
            if (pixel[0] !== pixel[1] || pixel[0] !== pixel[2]) {
                return false;
            }
        }
    }
    return true;
}