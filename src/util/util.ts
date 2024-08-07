import Point from '@mapbox/point-geometry';
import UnitBezier from '@mapbox/unitbezier';
import {isOffscreenCanvasDistorted} from './offscreen_canvas_distorted';
import type {Size} from './image';
import type {WorkerGlobalScopeInterface} from './web_worker';

/**
 * For a given collection of 2D points, returns their axis-aligned bounding box,
 * in the format [minX, minY, maxX, maxY].
 */
export function getAABB(points: Array<Point>): [number, number, number, number] {
    let tlX = Infinity;
    let tlY = Infinity;
    let brX = -Infinity;
    let brY = -Infinity;

    for (const p of points) {
        tlX = Math.min(tlX, p.x);
        tlY = Math.min(tlY, p.y);
        brX = Math.max(brX, p.x);
        brY = Math.max(brY, p.y);
    }

    return [tlX, tlY, brX, brY];
}

/**
 * Given a value `t` that varies between 0 and 1, return
 * an interpolation function that eases between 0 and 1 in a pleasing
 * cubic in-out fashion.
 */
export function easeCubicInOut(t: number): number {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    const t2 = t * t,
        t3 = t2 * t;
    return 4 * (t < 0.5 ? t3 : 3 * (t - t2) + t3 - 0.75);
}

/**
 * Given given (x, y), (x1, y1) control points for a bezier curve,
 * return a function that interpolates along that curve.
 *
 * @param p1x - control point 1 x coordinate
 * @param p1y - control point 1 y coordinate
 * @param p2x - control point 2 x coordinate
 * @param p2y - control point 2 y coordinate
 */
export function bezier(p1x: number, p1y: number, p2x: number, p2y: number): (t: number) => number {
    const bezier = new UnitBezier(p1x, p1y, p2x, p2y);
    return (t: number) => {
        return bezier.solve(t);
    };
}

/**
 * A default bezier-curve powered easing function with
 * control points (0.25, 0.1) and (0.25, 1)
 */
export const defaultEasing = bezier(0.25, 0.1, 0.25, 1);

/**
 * constrain n to the given range via min + max
 *
 * @param n - value
 * @param min - the minimum value to be returned
 * @param max - the maximum value to be returned
 * @returns the clamped value
 */
export function clamp(n: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, n));
}

/**
 * constrain n to the given range, excluding the minimum, via modular arithmetic
 *
 * @param n - value
 * @param min - the minimum value to be returned, exclusive
 * @param max - the maximum value to be returned, inclusive
 * @returns constrained number
 */
export function wrap(n: number, min: number, max: number): number {
    const d = max - min;
    const w = ((n - min) % d + d) % d + min;
    return (w === min) ? max : w;
}

/**
 * Compute the difference between the keys in one object and the keys
 * in another object.
 *
 * @returns keys difference
 */
export function keysDifference<S, T>(
    obj: {[key: string]: S},
    other: {[key: string]: T}
): Array<string> {
    const difference = [];
    for (const i in obj) {
        if (!(i in other)) {
            difference.push(i);
        }
    }
    return difference;
}

/**
 * Given a destination object and optionally many source objects,
 * copy all properties from the source objects into the destination.
 * The last source object given overrides properties from previous
 * source objects.
 *
 * @param dest - destination object
 * @param sources - sources from which properties are pulled
 */
export function extend<T extends {}, U>(dest: T, source: U): T & U;
export function extend<T extends {}, U, V>(dest: T, source1: U, source2: V): T & U & V;
export function extend<T extends {}, U, V, W>(dest: T, source1: U, source2: V, source3: W): T & U & V & W;
export function extend(dest: object, ...sources: Array<any>): any;
export function extend(dest: object, ...sources: Array<any>): any {
    for (const src of sources) {
        for (const k in src) {
            dest[k] = src[k];
        }
    }
    return dest;
}

// See https://stackoverflow.com/questions/49401866/all-possible-keys-of-an-union-type
type KeysOfUnion<T> = T extends T ? keyof T: never;

/**
 * Given an object and a number of properties as strings, return version
 * of that object with only those properties.
 *
 * @param src - the object
 * @param properties - an array of property names chosen
 * to appear on the resulting object.
 * @returns object with limited properties.
 * @example
 * ```ts
 * let foo = { name: 'Charlie', age: 10 };
 * let justName = pick(foo, ['name']); // justName = { name: 'Charlie' }
 * ```
 */
export function pick<T extends object>(src: T, properties: Array<KeysOfUnion<T>>): Partial<T> {
    const result: Partial<T> = {};
    for (let i = 0; i < properties.length; i++) {
        const k = properties[i];
        if (k in src) {
            result[k] = src[k];
        }
    }
    return result;
}

let id = 1;

/**
 * Return a unique numeric id, starting at 1 and incrementing with
 * each call.
 *
 * @returns unique numeric id.
 */
export function uniqueId(): number {
    return id++;
}

/**
 * Return whether a given value is a power of two
 */
export function isPowerOfTwo(value: number): boolean {
    return (Math.log(value) / Math.LN2) % 1 === 0;
}

/**
 * Return the next power of two, or the input value if already a power of two
 */
export function nextPowerOfTwo(value: number): number {
    if (value <= 1) return 1;
    return Math.pow(2, Math.ceil(Math.log(value) / Math.LN2));
}

/**
 * Create an object by mapping all the values of an existing object while
 * preserving their keys.
 */
export function mapObject(input: any, iterator: Function, context?: any): any {
    const output = {};
    for (const key in input) {
        output[key] = iterator.call(context || this, input[key], key, input);
    }
    return output;
}

/**
 * Create an object by filtering out values of an existing object.
 */
export function filterObject(input: any, iterator: Function, context?: any): any {
    const output = {};
    for (const key in input) {
        if (iterator.call(context || this, input[key], key, input)) {
            output[key] = input[key];
        }
    }
    return output;
}

/**
 * Deeply compares two object literals.
 * @param a - first object literal to be compared
 * @param b - second object literal to be compared
 * @returns true if the two object literals are deeply equal, false otherwise
 */
export function deepEqual(a?: unknown | null, b?: unknown | null): boolean {
    if (Array.isArray(a)) {
        if (!Array.isArray(b) || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (!deepEqual(a[i], b[i])) return false;
        }
        return true;
    }
    if (typeof a === 'object' && a !== null && b !== null) {
        if (!(typeof b === 'object')) return false;
        const keys = Object.keys(a);
        if (keys.length !== Object.keys(b).length) return false;
        for (const key in a) {
            if (!deepEqual(a[key], b[key])) return false;
        }
        return true;
    }
    return a === b;
}

/**
 * Deeply clones two objects.
 */
export function clone<T>(input: T): T {
    if (Array.isArray(input)) {
        return input.map(clone) as any as T;
    } else if (typeof input === 'object' && input) {
        return mapObject(input, clone) as any as T;
    } else {
        return input;
    }
}

/**
 * Check if two arrays have at least one common element.
 */
export function arraysIntersect<T>(a: Array<T>, b: Array<T>): boolean {
    for (let l = 0; l < a.length; l++) {
        if (b.indexOf(a[l]) >= 0) return true;
    }
    return false;
}

/**
 * Print a warning message to the console and ensure duplicate warning messages
 * are not printed.
 */
const warnOnceHistory: {[key: string]: boolean} = {};

export function warnOnce(message: string): void {
    if (!warnOnceHistory[message]) {
        // console isn't defined in some WebWorkers, see #2558
        if (typeof console !== 'undefined') console.warn(message);
        warnOnceHistory[message] = true;
    }
}

/**
 * Indicates if the provided Points are in a counter clockwise (true) or clockwise (false) order
 *
 * @returns true for a counter clockwise set of points
 */
// https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
export function isCounterClockwise(a: Point, b: Point, c: Point): boolean {
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
}

/**
 * For two lines a and b in 2d space, defined by any two points along the lines,
 * find the intersection point, or return null if the lines are parallel
 *
 * @param a1 - First point on line a
 * @param a2 - Second point on line a
 * @param b1 - First point on line b
 * @param b2 - Second point on line b
 *
 * @returns the intersection point of the two lines or null if they are parallel
 */
export function findLineIntersection(a1: Point, a2: Point, b1: Point, b2: Point): Point | null {
    const aDeltaY = a2.y - a1.y;
    const aDeltaX = a2.x - a1.x;
    const bDeltaY = b2.y - b1.y;
    const bDeltaX = b2.x - b1.x;

    const denominator = (bDeltaY * aDeltaX) - (bDeltaX * aDeltaY);

    if (denominator === 0) {
        // Lines are parallel
        return null;
    }

    const originDeltaY = a1.y - b1.y;
    const originDeltaX = a1.x - b1.x;
    const aInterpolation = (bDeltaX * originDeltaY - bDeltaY * originDeltaX) / denominator;

    // Find intersection by projecting out from origin of first segment
    return new Point(a1.x + (aInterpolation * aDeltaX), a1.y + (aInterpolation * aDeltaY));
}

/**
 * Converts spherical coordinates to cartesian coordinates.
 *
 * @param spherical - Spherical coordinates, in [radial, azimuthal, polar]
 * @returns cartesian coordinates in [x, y, z]
 */

export function sphericalToCartesian([r, azimuthal, polar]: [number, number, number]): {
    x: number;
    y: number;
    z: number;
} {
    // We abstract "north"/"up" (compass-wise) to be 0° when really this is 90° (π/2):
    // correct for that here
    azimuthal += 90;

    // Convert azimuthal and polar angles to radians
    azimuthal *= Math.PI / 180;
    polar *= Math.PI / 180;

    return {
        x: r * Math.cos(azimuthal) * Math.sin(polar),
        y: r * Math.sin(azimuthal) * Math.sin(polar),
        z: r * Math.cos(polar)
    };
}

/**
 *  Returns true if the when run in the web-worker context.
 *
 * @returns `true` if the when run in the web-worker context.
 */
export function isWorker(self: any): self is WorkerGlobalScopeInterface {
    // @ts-ignore
    return typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope;
}

/**
 * Parses data from 'Cache-Control' headers.
 *
 * @param cacheControl - Value of 'Cache-Control' header
 * @returns object containing parsed header info.
 */

export function parseCacheControl(cacheControl: string): any {
    // Taken from [Wreck](https://github.com/hapijs/wreck)
    const re = /(?:^|(?:\s*\,\s*))([^\x00-\x20\(\)<>@\,;\:\\"\/\[\]\?\=\{\}\x7F]+)(?:\=(?:([^\x00-\x20\(\)<>@\,;\:\\"\/\[\]\?\=\{\}\x7F]+)|(?:\"((?:[^"\\]|\\.)*)\")))?/g;

    const header = {};
    cacheControl.replace(re, ($0, $1, $2, $3) => {
        const value = $2 || $3;
        header[$1] = value ? value.toLowerCase() : true;
        return '';
    });

    if (header['max-age']) {
        const maxAge = parseInt(header['max-age'], 10);
        if (isNaN(maxAge)) delete header['max-age'];
        else header['max-age'] = maxAge;
    }

    return header;
}

let _isSafari = null;

/**
 * Returns true when run in WebKit derived browsers.
 * This is used as a workaround for a memory leak in Safari caused by using Transferable objects to
 * transfer data between WebWorkers and the main thread.
 * https://github.com/mapbox/mapbox-gl-js/issues/8771
 *
 * This should be removed once the underlying Safari issue is fixed.
 *
 * @param scope - Since this function is used both on the main thread and WebWorker context,
 *      let the calling scope pass in the global scope object.
 * @returns `true` when run in WebKit derived browsers.
 */
export function isSafari(scope: any): boolean {
    if (_isSafari == null) {
        const userAgent = scope.navigator ? scope.navigator.userAgent : null;
        _isSafari = !!scope.safari ||
        !!(userAgent && (/\b(iPad|iPhone|iPod)\b/.test(userAgent) || (!!userAgent.match('Safari') && !userAgent.match('Chrome'))));
    }
    return _isSafari;
}

export function storageAvailable(type: string): boolean {
    try {
        const storage = window[type];
        storage.setItem('_mapbox_test_', 1);
        storage.removeItem('_mapbox_test_');
        return true;
    } catch (e) {
        return false;
    }
}

// The following methods are from https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding#The_Unicode_Problem
//Unicode compliant base64 encoder for strings
export function b64EncodeUnicode(str: string) {
    return btoa(
        encodeURIComponent(str).replace(/%([0-9A-F]{2})/g,
            (match, p1) => {
                return String.fromCharCode(Number('0x' + p1)); //eslint-disable-line
            }
        )
    );
}

// Unicode compliant decoder for base64-encoded strings
export function b64DecodeUnicode(str: string) {
    return decodeURIComponent(atob(str).split('').map((c) => {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2); //eslint-disable-line
    }).join(''));
}

export function isImageBitmap(image: any): image is ImageBitmap {
    return typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap;
}

/**
 * Converts an ArrayBuffer to an ImageBitmap.
 *
 * Used mostly for testing purposes only, because mocking libs don't know how to work with ArrayBuffers, but work
 * perfectly fine with ImageBitmaps. Might also be used for environments (other than testing) not supporting
 * ArrayBuffers.
 *
 * @param data - Data to convert
 * @returns - A  promise resolved when the conversion is finished
 */
export const arrayBufferToImageBitmap = async (data: ArrayBuffer): Promise<ImageBitmap> => {
    if (data.byteLength === 0) {
        return createImageBitmap(new ImageData(1, 1));
    }
    const blob: Blob = new Blob([new Uint8Array(data)], {type: 'image/png'});
    try {
        return createImageBitmap(blob);
    } catch (e) {
        throw new Error(`Could not load image because of ${e.message}. Please make sure to use a supported image type such as PNG or JPEG. Note that SVGs are not supported.`);
    }
};

const transparentPngUrl = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQYV2NgAAIAAAUAAarVyFEAAAAASUVORK5CYII=';

/**
 * Converts an ArrayBuffer to an HTMLImageElement.
 *
 * Used mostly for testing purposes only, because mocking libs don't know how to work with ArrayBuffers, but work
 * perfectly fine with ImageBitmaps. Might also be used for environments (other than testing) not supporting
 * ArrayBuffers.
 *
 * @param data - Data to convert
 * @returns - A promise resolved when the conversion is finished
 */
export const arrayBufferToImage = (data: ArrayBuffer): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const img: HTMLImageElement = new Image();
        img.onload = () => {
            resolve(img);
            URL.revokeObjectURL(img.src);
            // prevent image dataURI memory leak in Safari;
            // but don't free the image immediately because it might be uploaded in the next frame
            // https://github.com/mapbox/mapbox-gl-js/issues/10226
            img.onload = null;
            window.requestAnimationFrame(() => { img.src = transparentPngUrl; });
        };
        img.onerror = () => reject(new Error('Could not load image. Please make sure to use a supported image type such as PNG or JPEG. Note that SVGs are not supported.'));
        const blob: Blob = new Blob([new Uint8Array(data)], {type: 'image/png'});
        img.src = data.byteLength ? URL.createObjectURL(blob) : transparentPngUrl;
    });
};

/**
 * Computes the webcodecs VideoFrame API options to select a rectangle out of
 * an image and write it into the destination rectangle.
 *
 * Rect (x/y/width/height) select the overlapping rectangle from the source image
 * and layout (offset/stride) write that overlapping rectangle to the correct place
 * in the destination image.
 *
 * Offset is the byte offset in the dest image that the first pixel appears at
 * and stride is the number of bytes to the start of the next row:
 * ┌───────────┐
 * │  dest     │
 * │       ┌───┼───────┐
 * │offset→│▓▓▓│ source│
 * │       │▓▓▓│       │
 * │       └───┼───────┘
 * │stride ⇠╌╌╌│
 * │╌╌╌╌╌╌→    │
 * └───────────┘
 *
 * @param image - source image containing a width and height attribute
 * @param x - top-left x coordinate to read from the image
 * @param y - top-left y coordinate to read from the image
 * @param width - width of the rectangle to read from the image
 * @param height - height of the rectangle to read from the image
 * @returns the layout and rect options to pass into VideoFrame API
 */
function computeVideoFrameParameters(image: Size, x: number, y: number, width: number, height: number): VideoFrameCopyToOptions {
    const destRowOffset = Math.max(-x, 0) * 4;
    const firstSourceRow = Math.max(0, y);
    const firstDestRow = firstSourceRow - y;
    const offset = firstDestRow * width * 4 + destRowOffset;
    const stride = width * 4;

    const sourceLeft = Math.max(0, x);
    const sourceTop = Math.max(0, y);
    const sourceRight = Math.min(image.width, x + width);
    const sourceBottom = Math.min(image.height, y + height);
    return {
        rect: {
            x: sourceLeft,
            y: sourceTop,
            width: sourceRight - sourceLeft,
            height: sourceBottom - sourceTop
        },
        layout: [{offset, stride}]
    };
}

/**
 * Reads pixels from an ImageBitmap/Image/canvas using webcodec VideoFrame API.
 *
 * @param data - image, imagebitmap, or canvas to parse
 * @param x - top-left x coordinate to read from the image
 * @param y - top-left y coordinate to read from the image
 * @param width - width of the rectangle to read from the image
 * @param height - height of the rectangle to read from the image
 * @returns a promise containing the parsed RGBA pixel values of the image, or the error if an error occurred
 */
export async function readImageUsingVideoFrame(
    image: HTMLImageElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas,
    x: number, y: number, width: number, height: number
): Promise<Uint8ClampedArray> {
    if (typeof VideoFrame === 'undefined') {
        throw new Error('VideoFrame not supported');
    }
    const frame = new VideoFrame(image, {timestamp: 0});
    try {
        const format = frame?.format;
        if (!format || !(format.startsWith('BGR') || format.startsWith('RGB'))) {
            throw new Error(`Unrecognized format ${format}`);
        }
        const swapBR = format.startsWith('BGR');
        const result = new Uint8ClampedArray(width * height * 4);
        await frame.copyTo(result, computeVideoFrameParameters(image, x, y, width, height));
        if (swapBR) {
            for (let i = 0; i < result.length; i += 4) {
                const tmp = result[i];
                result[i] = result[i + 2];
                result[i + 2] = tmp;
            }
        }
        return result;
    } finally {
        frame.close();
    }
}

let offscreenCanvas: OffscreenCanvas;
let offscreenCanvasContext: OffscreenCanvasRenderingContext2D;

/**
 * Reads pixels from an ImageBitmap/Image/canvas using OffscreenCanvas
 *
 * @param data - image, imagebitmap, or canvas to parse
 * @param x - top-left x coordinate to read from the image
 * @param y - top-left y coordinate to read from the image
 * @param width - width of the rectangle to read from the image
 * @param height - height of the rectangle to read from the image
 * @returns a promise containing the parsed RGBA pixel values of the image, or the error if an error occurred
 */
export function readImageDataUsingOffscreenCanvas(
    imgBitmap: HTMLImageElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas,
    x: number, y: number, width: number, height: number
): Uint8ClampedArray {
    const origWidth = imgBitmap.width;
    const origHeight = imgBitmap.height;
    // Lazily initialize OffscreenCanvas
    if (!offscreenCanvas || !offscreenCanvasContext) {
        // Dem tiles are typically 256x256
        offscreenCanvas = new OffscreenCanvas(origWidth, origHeight);
        offscreenCanvasContext = offscreenCanvas.getContext('2d', {willReadFrequently: true});
    }

    offscreenCanvas.width = origWidth;
    offscreenCanvas.height = origHeight;

    offscreenCanvasContext.drawImage(imgBitmap, 0, 0, origWidth, origHeight);
    const imgData = offscreenCanvasContext.getImageData(x, y, width, height);
    offscreenCanvasContext.clearRect(0, 0, origWidth, origHeight);
    return imgData.data;
}

/**
 * Reads RGBA pixels from an preferring OffscreenCanvas, but falling back to VideoFrame if supported and
 * the browser is mangling OffscreenCanvas getImageData results.
 *
 * @param data - image, imagebitmap, or canvas to parse
 * @param x - top-left x coordinate to read from the image
 * @param y - top-left y coordinate to read from the image
 * @param width - width of the rectangle to read from the image
 * @param height - height of the rectangle to read from the image
 * @returns a promise containing the parsed RGBA pixel values of the image
 */
export async function getImageData(
    image: HTMLImageElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas,
    x: number, y: number, width: number, height: number
): Promise<Uint8ClampedArray> {
    if (isOffscreenCanvasDistorted()) {
        try {
            return await readImageUsingVideoFrame(image, x, y, width, height);
        } catch (e) {
            // fall back to OffscreenCanvas
        }
    }
    return readImageDataUsingOffscreenCanvas(image, x, y, width, height);
}

export interface Subscription {
    unsubscribe(): void;
}

export interface Subscriber {
    addEventListener: typeof window.addEventListener;
    removeEventListener: typeof window.removeEventListener;
}

/**
 * This method is used in order to register an event listener using a lambda function.
 * The return value will allow unsubscribing from the event, without the need to store the method reference.
 * @param target - The target
 * @param message - The message
 * @param listener - The listener
 * @param options - The options
 * @returns a subscription object that can be used to unsubscribe from the event
 */
export function subscribe(target: Subscriber, message: keyof WindowEventMap, listener: (...args: any) => void, options: boolean | AddEventListenerOptions): Subscription {
    target.addEventListener(message, listener, options);
    return {
        unsubscribe: () => {
            target.removeEventListener(message, listener, options);
        }
    };
}

/**
 * This method converts degrees to radians.
 * The return value is the radian value.
 * @param degrees - The number of degrees
 * @returns radians
 */
export function degreesToRadians(degrees: number): number {
    return degrees * Math.PI / 180;
}

/**
 * Makes optional keys required and add the the undefined type.
 *
 * ```
 * interface Test {
 *  foo: number;
 *  bar?: number;
 *  baz: number | undefined;
 * }
 *
 * Complete<Test> {
 *  foo: number;
 *  bar: number | undefined;
 *  baz: number | undefined;
 * }
 *
 * ```
 *
 * See https://medium.com/terria/typescript-transforming-optional-properties-to-required-properties-that-may-be-undefined-7482cb4e1585
 */

export type Complete<T> = {
    [P in keyof Required<T>]: Pick<T, P> extends Required<Pick<T, P>> ? T[P] : (T[P] | undefined);
}

export type TileJSON = {
    tilejson: '2.2.0' | '2.1.0' | '2.0.1' | '2.0.0' | '1.0.0';
    name?: string;
    description?: string;
    version?: string;
    attribution?: string;
    template?: string;
    tiles: Array<string>;
    grids?: Array<string>;
    data?: Array<string>;
    minzoom?: number;
    maxzoom?: number;
    bounds?: [number, number, number, number];
    center?: [number, number, number];
    vector_layers: [{id: string}]; // this is partial but enough for what we need
};
  
function splitTextByGraphemesWithIntl(text: string): string[] {  
    const segmenter = new Intl.Segmenter();
    const segments = Array.from(segmenter.segment(text));  
    const ret = segments.map(segment => segment.segment);
    return ret;
}

const RAW_PROPERTY_LIST = `
# GraphemeBreakProperty-15.1.0.txt
# Date: 2023-01-05, 20:34:41 GMT
# © 2023 Unicode®, Inc.
# Unicode and the Unicode Logo are registered trademarks of Unicode, Inc. in the U.S. and other countries.
# For terms of use, see https://www.unicode.org/terms_of_use.html
#
# Unicode Character Database
#   For documentation, see https://www.unicode.org/reports/tr44/

# ================================================

# Property:	Grapheme_Cluster_Break

#  All code points not explicitly listed for Grapheme_Cluster_Break
#  have the value Other (XX).

# @missing: 0000..10FFFF; Other

# ================================================

0600..0605    ; Prepend # Cf   [6] ARABIC NUMBER SIGN..ARABIC NUMBER MARK ABOVE
06DD          ; Prepend # Cf       ARABIC END OF AYAH
070F          ; Prepend # Cf       SYRIAC ABBREVIATION MARK
0890..0891    ; Prepend # Cf   [2] ARABIC POUND MARK ABOVE..ARABIC PIASTRE MARK ABOVE
08E2          ; Prepend # Cf       ARABIC DISPUTED END OF AYAH
0D4E          ; Prepend # Lo       MALAYALAM LETTER DOT REPH
110BD         ; Prepend # Cf       KAITHI NUMBER SIGN
110CD         ; Prepend # Cf       KAITHI NUMBER SIGN ABOVE
111C2..111C3  ; Prepend # Lo   [2] SHARADA SIGN JIHVAMULIYA..SHARADA SIGN UPADHMANIYA
1193F         ; Prepend # Lo       DIVES AKURU PREFIXED NASAL SIGN
11941         ; Prepend # Lo       DIVES AKURU INITIAL RA
11A3A         ; Prepend # Lo       ZANABAZAR SQUARE CLUSTER-INITIAL LETTER RA
11A84..11A89  ; Prepend # Lo   [6] SOYOMBO SIGN JIHVAMULIYA..SOYOMBO CLUSTER-INITIAL LETTER SA
11D46         ; Prepend # Lo       MASARAM GONDI REPHA
11F02         ; Prepend # Lo       KAWI SIGN REPHA

# Total code points: 27

# ================================================

000D          ; CR # Cc       <control-000D>

# Total code points: 1

# ================================================

000A          ; LF # Cc       <control-000A>

# Total code points: 1

# ================================================

0000..0009    ; Control # Cc  [10] <control-0000>..<control-0009>
000B..000C    ; Control # Cc   [2] <control-000B>..<control-000C>
000E..001F    ; Control # Cc  [18] <control-000E>..<control-001F>
007F..009F    ; Control # Cc  [33] <control-007F>..<control-009F>
00AD          ; Control # Cf       SOFT HYPHEN
061C          ; Control # Cf       ARABIC LETTER MARK
180E          ; Control # Cf       MONGOLIAN VOWEL SEPARATOR
200B          ; Control # Cf       ZERO WIDTH SPACE
200E..200F    ; Control # Cf   [2] LEFT-TO-RIGHT MARK..RIGHT-TO-LEFT MARK
2028          ; Control # Zl       LINE SEPARATOR
2029          ; Control # Zp       PARAGRAPH SEPARATOR
202A..202E    ; Control # Cf   [5] LEFT-TO-RIGHT EMBEDDING..RIGHT-TO-LEFT OVERRIDE
2060..2064    ; Control # Cf   [5] WORD JOINER..INVISIBLE PLUS
2065          ; Control # Cn       <reserved-2065>
2066..206F    ; Control # Cf  [10] LEFT-TO-RIGHT ISOLATE..NOMINAL DIGIT SHAPES
FEFF          ; Control # Cf       ZERO WIDTH NO-BREAK SPACE
FFF0..FFF8    ; Control # Cn   [9] <reserved-FFF0>..<reserved-FFF8>
FFF9..FFFB    ; Control # Cf   [3] INTERLINEAR ANNOTATION ANCHOR..INTERLINEAR ANNOTATION TERMINATOR
13430..1343F  ; Control # Cf  [16] EGYPTIAN HIEROGLYPH VERTICAL JOINER..EGYPTIAN HIEROGLYPH END WALLED ENCLOSURE
1BCA0..1BCA3  ; Control # Cf   [4] SHORTHAND FORMAT LETTER OVERLAP..SHORTHAND FORMAT UP STEP
1D173..1D17A  ; Control # Cf   [8] MUSICAL SYMBOL BEGIN BEAM..MUSICAL SYMBOL END PHRASE
E0000         ; Control # Cn       <reserved-E0000>
E0001         ; Control # Cf       LANGUAGE TAG
E0002..E001F  ; Control # Cn  [30] <reserved-E0002>..<reserved-E001F>
E0080..E00FF  ; Control # Cn [128] <reserved-E0080>..<reserved-E00FF>
E01F0..E0FFF  ; Control # Cn [3600] <reserved-E01F0>..<reserved-E0FFF>

# Total code points: 3893

# ================================================

0300..036F    ; Extend # Mn [112] COMBINING GRAVE ACCENT..COMBINING LATIN SMALL LETTER X
0483..0487    ; Extend # Mn   [5] COMBINING CYRILLIC TITLO..COMBINING CYRILLIC POKRYTIE
0488..0489    ; Extend # Me   [2] COMBINING CYRILLIC HUNDRED THOUSANDS SIGN..COMBINING CYRILLIC MILLIONS SIGN
0591..05BD    ; Extend # Mn  [45] HEBREW ACCENT ETNAHTA..HEBREW POINT METEG
05BF          ; Extend # Mn       HEBREW POINT RAFE
05C1..05C2    ; Extend # Mn   [2] HEBREW POINT SHIN DOT..HEBREW POINT SIN DOT
05C4..05C5    ; Extend # Mn   [2] HEBREW MARK UPPER DOT..HEBREW MARK LOWER DOT
05C7          ; Extend # Mn       HEBREW POINT QAMATS QATAN
0610..061A    ; Extend # Mn  [11] ARABIC SIGN SALLALLAHOU ALAYHE WASSALLAM..ARABIC SMALL KASRA
064B..065F    ; Extend # Mn  [21] ARABIC FATHATAN..ARABIC WAVY HAMZA BELOW
0670          ; Extend # Mn       ARABIC LETTER SUPERSCRIPT ALEF
06D6..06DC    ; Extend # Mn   [7] ARABIC SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA..ARABIC SMALL HIGH SEEN
06DF..06E4    ; Extend # Mn   [6] ARABIC SMALL HIGH ROUNDED ZERO..ARABIC SMALL HIGH MADDA
06E7..06E8    ; Extend # Mn   [2] ARABIC SMALL HIGH YEH..ARABIC SMALL HIGH NOON
06EA..06ED    ; Extend # Mn   [4] ARABIC EMPTY CENTRE LOW STOP..ARABIC SMALL LOW MEEM
0711          ; Extend # Mn       SYRIAC LETTER SUPERSCRIPT ALAPH
0730..074A    ; Extend # Mn  [27] SYRIAC PTHAHA ABOVE..SYRIAC BARREKH
07A6..07B0    ; Extend # Mn  [11] THAANA ABAFILI..THAANA SUKUN
07EB..07F3    ; Extend # Mn   [9] NKO COMBINING SHORT HIGH TONE..NKO COMBINING DOUBLE DOT ABOVE
07FD          ; Extend # Mn       NKO DANTAYALAN
0816..0819    ; Extend # Mn   [4] SAMARITAN MARK IN..SAMARITAN MARK DAGESH
081B..0823    ; Extend # Mn   [9] SAMARITAN MARK EPENTHETIC YUT..SAMARITAN VOWEL SIGN A
0825..0827    ; Extend # Mn   [3] SAMARITAN VOWEL SIGN SHORT A..SAMARITAN VOWEL SIGN U
0829..082D    ; Extend # Mn   [5] SAMARITAN VOWEL SIGN LONG I..SAMARITAN MARK NEQUDAA
0859..085B    ; Extend # Mn   [3] MANDAIC AFFRICATION MARK..MANDAIC GEMINATION MARK
0898..089F    ; Extend # Mn   [8] ARABIC SMALL HIGH WORD AL-JUZ..ARABIC HALF MADDA OVER MADDA
08CA..08E1    ; Extend # Mn  [24] ARABIC SMALL HIGH FARSI YEH..ARABIC SMALL HIGH SIGN SAFHA
08E3..0902    ; Extend # Mn  [32] ARABIC TURNED DAMMA BELOW..DEVANAGARI SIGN ANUSVARA
093A          ; Extend # Mn       DEVANAGARI VOWEL SIGN OE
093C          ; Extend # Mn       DEVANAGARI SIGN NUKTA
0941..0948    ; Extend # Mn   [8] DEVANAGARI VOWEL SIGN U..DEVANAGARI VOWEL SIGN AI
094D          ; Extend # Mn       DEVANAGARI SIGN VIRAMA
0951..0957    ; Extend # Mn   [7] DEVANAGARI STRESS SIGN UDATTA..DEVANAGARI VOWEL SIGN UUE
0962..0963    ; Extend # Mn   [2] DEVANAGARI VOWEL SIGN VOCALIC L..DEVANAGARI VOWEL SIGN VOCALIC LL
0981          ; Extend # Mn       BENGALI SIGN CANDRABINDU
09BC          ; Extend # Mn       BENGALI SIGN NUKTA
09BE          ; Extend # Mc       BENGALI VOWEL SIGN AA
09C1..09C4    ; Extend # Mn   [4] BENGALI VOWEL SIGN U..BENGALI VOWEL SIGN VOCALIC RR
09CD          ; Extend # Mn       BENGALI SIGN VIRAMA
09D7          ; Extend # Mc       BENGALI AU LENGTH MARK
09E2..09E3    ; Extend # Mn   [2] BENGALI VOWEL SIGN VOCALIC L..BENGALI VOWEL SIGN VOCALIC LL
09FE          ; Extend # Mn       BENGALI SANDHI MARK
0A01..0A02    ; Extend # Mn   [2] GURMUKHI SIGN ADAK BINDI..GURMUKHI SIGN BINDI
0A3C          ; Extend # Mn       GURMUKHI SIGN NUKTA
0A41..0A42    ; Extend # Mn   [2] GURMUKHI VOWEL SIGN U..GURMUKHI VOWEL SIGN UU
0A47..0A48    ; Extend # Mn   [2] GURMUKHI VOWEL SIGN EE..GURMUKHI VOWEL SIGN AI
0A4B..0A4D    ; Extend # Mn   [3] GURMUKHI VOWEL SIGN OO..GURMUKHI SIGN VIRAMA
0A51          ; Extend # Mn       GURMUKHI SIGN UDAAT
0A70..0A71    ; Extend # Mn   [2] GURMUKHI TIPPI..GURMUKHI ADDAK
0A75          ; Extend # Mn       GURMUKHI SIGN YAKASH
0A81..0A82    ; Extend # Mn   [2] GUJARATI SIGN CANDRABINDU..GUJARATI SIGN ANUSVARA
0ABC          ; Extend # Mn       GUJARATI SIGN NUKTA
0AC1..0AC5    ; Extend # Mn   [5] GUJARATI VOWEL SIGN U..GUJARATI VOWEL SIGN CANDRA E
0AC7..0AC8    ; Extend # Mn   [2] GUJARATI VOWEL SIGN E..GUJARATI VOWEL SIGN AI
0ACD          ; Extend # Mn       GUJARATI SIGN VIRAMA
0AE2..0AE3    ; Extend # Mn   [2] GUJARATI VOWEL SIGN VOCALIC L..GUJARATI VOWEL SIGN VOCALIC LL
0AFA..0AFF    ; Extend # Mn   [6] GUJARATI SIGN SUKUN..GUJARATI SIGN TWO-CIRCLE NUKTA ABOVE
0B01          ; Extend # Mn       ORIYA SIGN CANDRABINDU
0B3C          ; Extend # Mn       ORIYA SIGN NUKTA
0B3E          ; Extend # Mc       ORIYA VOWEL SIGN AA
0B3F          ; Extend # Mn       ORIYA VOWEL SIGN I
0B41..0B44    ; Extend # Mn   [4] ORIYA VOWEL SIGN U..ORIYA VOWEL SIGN VOCALIC RR
0B4D          ; Extend # Mn       ORIYA SIGN VIRAMA
0B55..0B56    ; Extend # Mn   [2] ORIYA SIGN OVERLINE..ORIYA AI LENGTH MARK
0B57          ; Extend # Mc       ORIYA AU LENGTH MARK
0B62..0B63    ; Extend # Mn   [2] ORIYA VOWEL SIGN VOCALIC L..ORIYA VOWEL SIGN VOCALIC LL
0B82          ; Extend # Mn       TAMIL SIGN ANUSVARA
0BBE          ; Extend # Mc       TAMIL VOWEL SIGN AA
0BC0          ; Extend # Mn       TAMIL VOWEL SIGN II
0BCD          ; Extend # Mn       TAMIL SIGN VIRAMA
0BD7          ; Extend # Mc       TAMIL AU LENGTH MARK
0C00          ; Extend # Mn       TELUGU SIGN COMBINING CANDRABINDU ABOVE
0C04          ; Extend # Mn       TELUGU SIGN COMBINING ANUSVARA ABOVE
0C3C          ; Extend # Mn       TELUGU SIGN NUKTA
0C3E..0C40    ; Extend # Mn   [3] TELUGU VOWEL SIGN AA..TELUGU VOWEL SIGN II
0C46..0C48    ; Extend # Mn   [3] TELUGU VOWEL SIGN E..TELUGU VOWEL SIGN AI
0C4A..0C4D    ; Extend # Mn   [4] TELUGU VOWEL SIGN O..TELUGU SIGN VIRAMA
0C55..0C56    ; Extend # Mn   [2] TELUGU LENGTH MARK..TELUGU AI LENGTH MARK
0C62..0C63    ; Extend # Mn   [2] TELUGU VOWEL SIGN VOCALIC L..TELUGU VOWEL SIGN VOCALIC LL
0C81          ; Extend # Mn       KANNADA SIGN CANDRABINDU
0CBC          ; Extend # Mn       KANNADA SIGN NUKTA
0CBF          ; Extend # Mn       KANNADA VOWEL SIGN I
0CC2          ; Extend # Mc       KANNADA VOWEL SIGN UU
0CC6          ; Extend # Mn       KANNADA VOWEL SIGN E
0CCC..0CCD    ; Extend # Mn   [2] KANNADA VOWEL SIGN AU..KANNADA SIGN VIRAMA
0CD5..0CD6    ; Extend # Mc   [2] KANNADA LENGTH MARK..KANNADA AI LENGTH MARK
0CE2..0CE3    ; Extend # Mn   [2] KANNADA VOWEL SIGN VOCALIC L..KANNADA VOWEL SIGN VOCALIC LL
0D00..0D01    ; Extend # Mn   [2] MALAYALAM SIGN COMBINING ANUSVARA ABOVE..MALAYALAM SIGN CANDRABINDU
0D3B..0D3C    ; Extend # Mn   [2] MALAYALAM SIGN VERTICAL BAR VIRAMA..MALAYALAM SIGN CIRCULAR VIRAMA
0D3E          ; Extend # Mc       MALAYALAM VOWEL SIGN AA
0D41..0D44    ; Extend # Mn   [4] MALAYALAM VOWEL SIGN U..MALAYALAM VOWEL SIGN VOCALIC RR
0D4D          ; Extend # Mn       MALAYALAM SIGN VIRAMA
0D57          ; Extend # Mc       MALAYALAM AU LENGTH MARK
0D62..0D63    ; Extend # Mn   [2] MALAYALAM VOWEL SIGN VOCALIC L..MALAYALAM VOWEL SIGN VOCALIC LL
0D81          ; Extend # Mn       SINHALA SIGN CANDRABINDU
0DCA          ; Extend # Mn       SINHALA SIGN AL-LAKUNA
0DCF          ; Extend # Mc       SINHALA VOWEL SIGN AELA-PILLA
0DD2..0DD4    ; Extend # Mn   [3] SINHALA VOWEL SIGN KETTI IS-PILLA..SINHALA VOWEL SIGN KETTI PAA-PILLA
0DD6          ; Extend # Mn       SINHALA VOWEL SIGN DIGA PAA-PILLA
0DDF          ; Extend # Mc       SINHALA VOWEL SIGN GAYANUKITTA
0E31          ; Extend # Mn       THAI CHARACTER MAI HAN-AKAT
0E34..0E3A    ; Extend # Mn   [7] THAI CHARACTER SARA I..THAI CHARACTER PHINTHU
0E47..0E4E    ; Extend # Mn   [8] THAI CHARACTER MAITAIKHU..THAI CHARACTER YAMAKKAN
0EB1          ; Extend # Mn       LAO VOWEL SIGN MAI KAN
0EB4..0EBC    ; Extend # Mn   [9] LAO VOWEL SIGN I..LAO SEMIVOWEL SIGN LO
0EC8..0ECE    ; Extend # Mn   [7] LAO TONE MAI EK..LAO YAMAKKAN
0F18..0F19    ; Extend # Mn   [2] TIBETAN ASTROLOGICAL SIGN -KHYUD PA..TIBETAN ASTROLOGICAL SIGN SDONG TSHUGS
0F35          ; Extend # Mn       TIBETAN MARK NGAS BZUNG NYI ZLA
0F37          ; Extend # Mn       TIBETAN MARK NGAS BZUNG SGOR RTAGS
0F39          ; Extend # Mn       TIBETAN MARK TSA -PHRU
0F71..0F7E    ; Extend # Mn  [14] TIBETAN VOWEL SIGN AA..TIBETAN SIGN RJES SU NGA RO
0F80..0F84    ; Extend # Mn   [5] TIBETAN VOWEL SIGN REVERSED I..TIBETAN MARK HALANTA
0F86..0F87    ; Extend # Mn   [2] TIBETAN SIGN LCI RTAGS..TIBETAN SIGN YANG RTAGS
0F8D..0F97    ; Extend # Mn  [11] TIBETAN SUBJOINED SIGN LCE TSA CAN..TIBETAN SUBJOINED LETTER JA
0F99..0FBC    ; Extend # Mn  [36] TIBETAN SUBJOINED LETTER NYA..TIBETAN SUBJOINED LETTER FIXED-FORM RA
0FC6          ; Extend # Mn       TIBETAN SYMBOL PADMA GDAN
102D..1030    ; Extend # Mn   [4] MYANMAR VOWEL SIGN I..MYANMAR VOWEL SIGN UU
1032..1037    ; Extend # Mn   [6] MYANMAR VOWEL SIGN AI..MYANMAR SIGN DOT BELOW
1039..103A    ; Extend # Mn   [2] MYANMAR SIGN VIRAMA..MYANMAR SIGN ASAT
103D..103E    ; Extend # Mn   [2] MYANMAR CONSONANT SIGN MEDIAL WA..MYANMAR CONSONANT SIGN MEDIAL HA
1058..1059    ; Extend # Mn   [2] MYANMAR VOWEL SIGN VOCALIC L..MYANMAR VOWEL SIGN VOCALIC LL
105E..1060    ; Extend # Mn   [3] MYANMAR CONSONANT SIGN MON MEDIAL NA..MYANMAR CONSONANT SIGN MON MEDIAL LA
1071..1074    ; Extend # Mn   [4] MYANMAR VOWEL SIGN GEBA KAREN I..MYANMAR VOWEL SIGN KAYAH EE
1082          ; Extend # Mn       MYANMAR CONSONANT SIGN SHAN MEDIAL WA
1085..1086    ; Extend # Mn   [2] MYANMAR VOWEL SIGN SHAN E ABOVE..MYANMAR VOWEL SIGN SHAN FINAL Y
108D          ; Extend # Mn       MYANMAR SIGN SHAN COUNCIL EMPHATIC TONE
109D          ; Extend # Mn       MYANMAR VOWEL SIGN AITON AI
135D..135F    ; Extend # Mn   [3] ETHIOPIC COMBINING GEMINATION AND VOWEL LENGTH MARK..ETHIOPIC COMBINING GEMINATION MARK
1712..1714    ; Extend # Mn   [3] TAGALOG VOWEL SIGN I..TAGALOG SIGN VIRAMA
1732..1733    ; Extend # Mn   [2] HANUNOO VOWEL SIGN I..HANUNOO VOWEL SIGN U
1752..1753    ; Extend # Mn   [2] BUHID VOWEL SIGN I..BUHID VOWEL SIGN U
1772..1773    ; Extend # Mn   [2] TAGBANWA VOWEL SIGN I..TAGBANWA VOWEL SIGN U
17B4..17B5    ; Extend # Mn   [2] KHMER VOWEL INHERENT AQ..KHMER VOWEL INHERENT AA
17B7..17BD    ; Extend # Mn   [7] KHMER VOWEL SIGN I..KHMER VOWEL SIGN UA
17C6          ; Extend # Mn       KHMER SIGN NIKAHIT
17C9..17D3    ; Extend # Mn  [11] KHMER SIGN MUUSIKATOAN..KHMER SIGN BATHAMASAT
17DD          ; Extend # Mn       KHMER SIGN ATTHACAN
180B..180D    ; Extend # Mn   [3] MONGOLIAN FREE VARIATION SELECTOR ONE..MONGOLIAN FREE VARIATION SELECTOR THREE
180F          ; Extend # Mn       MONGOLIAN FREE VARIATION SELECTOR FOUR
1885..1886    ; Extend # Mn   [2] MONGOLIAN LETTER ALI GALI BALUDA..MONGOLIAN LETTER ALI GALI THREE BALUDA
18A9          ; Extend # Mn       MONGOLIAN LETTER ALI GALI DAGALGA
1920..1922    ; Extend # Mn   [3] LIMBU VOWEL SIGN A..LIMBU VOWEL SIGN U
1927..1928    ; Extend # Mn   [2] LIMBU VOWEL SIGN E..LIMBU VOWEL SIGN O
1932          ; Extend # Mn       LIMBU SMALL LETTER ANUSVARA
1939..193B    ; Extend # Mn   [3] LIMBU SIGN MUKPHRENG..LIMBU SIGN SA-I
1A17..1A18    ; Extend # Mn   [2] BUGINESE VOWEL SIGN I..BUGINESE VOWEL SIGN U
1A1B          ; Extend # Mn       BUGINESE VOWEL SIGN AE
1A56          ; Extend # Mn       TAI THAM CONSONANT SIGN MEDIAL LA
1A58..1A5E    ; Extend # Mn   [7] TAI THAM SIGN MAI KANG LAI..TAI THAM CONSONANT SIGN SA
1A60          ; Extend # Mn       TAI THAM SIGN SAKOT
1A62          ; Extend # Mn       TAI THAM VOWEL SIGN MAI SAT
1A65..1A6C    ; Extend # Mn   [8] TAI THAM VOWEL SIGN I..TAI THAM VOWEL SIGN OA BELOW
1A73..1A7C    ; Extend # Mn  [10] TAI THAM VOWEL SIGN OA ABOVE..TAI THAM SIGN KHUEN-LUE KARAN
1A7F          ; Extend # Mn       TAI THAM COMBINING CRYPTOGRAMMIC DOT
1AB0..1ABD    ; Extend # Mn  [14] COMBINING DOUBLED CIRCUMFLEX ACCENT..COMBINING PARENTHESES BELOW
1ABE          ; Extend # Me       COMBINING PARENTHESES OVERLAY
1ABF..1ACE    ; Extend # Mn  [16] COMBINING LATIN SMALL LETTER W BELOW..COMBINING LATIN SMALL LETTER INSULAR T
1B00..1B03    ; Extend # Mn   [4] BALINESE SIGN ULU RICEM..BALINESE SIGN SURANG
1B34          ; Extend # Mn       BALINESE SIGN REREKAN
1B35          ; Extend # Mc       BALINESE VOWEL SIGN TEDUNG
1B36..1B3A    ; Extend # Mn   [5] BALINESE VOWEL SIGN ULU..BALINESE VOWEL SIGN RA REPA
1B3C          ; Extend # Mn       BALINESE VOWEL SIGN LA LENGA
1B42          ; Extend # Mn       BALINESE VOWEL SIGN PEPET
1B6B..1B73    ; Extend # Mn   [9] BALINESE MUSICAL SYMBOL COMBINING TEGEH..BALINESE MUSICAL SYMBOL COMBINING GONG
1B80..1B81    ; Extend # Mn   [2] SUNDANESE SIGN PANYECEK..SUNDANESE SIGN PANGLAYAR
1BA2..1BA5    ; Extend # Mn   [4] SUNDANESE CONSONANT SIGN PANYAKRA..SUNDANESE VOWEL SIGN PANYUKU
1BA8..1BA9    ; Extend # Mn   [2] SUNDANESE VOWEL SIGN PAMEPET..SUNDANESE VOWEL SIGN PANEULEUNG
1BAB..1BAD    ; Extend # Mn   [3] SUNDANESE SIGN VIRAMA..SUNDANESE CONSONANT SIGN PASANGAN WA
1BE6          ; Extend # Mn       BATAK SIGN TOMPI
1BE8..1BE9    ; Extend # Mn   [2] BATAK VOWEL SIGN PAKPAK E..BATAK VOWEL SIGN EE
1BED          ; Extend # Mn       BATAK VOWEL SIGN KARO O
1BEF..1BF1    ; Extend # Mn   [3] BATAK VOWEL SIGN U FOR SIMALUNGUN SA..BATAK CONSONANT SIGN H
1C2C..1C33    ; Extend # Mn   [8] LEPCHA VOWEL SIGN E..LEPCHA CONSONANT SIGN T
1C36..1C37    ; Extend # Mn   [2] LEPCHA SIGN RAN..LEPCHA SIGN NUKTA
1CD0..1CD2    ; Extend # Mn   [3] VEDIC TONE KARSHANA..VEDIC TONE PRENKHA
1CD4..1CE0    ; Extend # Mn  [13] VEDIC SIGN YAJURVEDIC MIDLINE SVARITA..VEDIC TONE RIGVEDIC KASHMIRI INDEPENDENT SVARITA
1CE2..1CE8    ; Extend # Mn   [7] VEDIC SIGN VISARGA SVARITA..VEDIC SIGN VISARGA ANUDATTA WITH TAIL
1CED          ; Extend # Mn       VEDIC SIGN TIRYAK
1CF4          ; Extend # Mn       VEDIC TONE CANDRA ABOVE
1CF8..1CF9    ; Extend # Mn   [2] VEDIC TONE RING ABOVE..VEDIC TONE DOUBLE RING ABOVE
1DC0..1DFF    ; Extend # Mn  [64] COMBINING DOTTED GRAVE ACCENT..COMBINING RIGHT ARROWHEAD AND DOWN ARROWHEAD BELOW
200C          ; Extend # Cf       ZERO WIDTH NON-JOINER
20D0..20DC    ; Extend # Mn  [13] COMBINING LEFT HARPOON ABOVE..COMBINING FOUR DOTS ABOVE
20DD..20E0    ; Extend # Me   [4] COMBINING ENCLOSING CIRCLE..COMBINING ENCLOSING CIRCLE BACKSLASH
20E1          ; Extend # Mn       COMBINING LEFT RIGHT ARROW ABOVE
20E2..20E4    ; Extend # Me   [3] COMBINING ENCLOSING SCREEN..COMBINING ENCLOSING UPWARD POINTING TRIANGLE
20E5..20F0    ; Extend # Mn  [12] COMBINING REVERSE SOLIDUS OVERLAY..COMBINING ASTERISK ABOVE
2CEF..2CF1    ; Extend # Mn   [3] COPTIC COMBINING NI ABOVE..COPTIC COMBINING SPIRITUS LENIS
2D7F          ; Extend # Mn       TIFINAGH CONSONANT JOINER
2DE0..2DFF    ; Extend # Mn  [32] COMBINING CYRILLIC LETTER BE..COMBINING CYRILLIC LETTER IOTIFIED BIG YUS
302A..302D    ; Extend # Mn   [4] IDEOGRAPHIC LEVEL TONE MARK..IDEOGRAPHIC ENTERING TONE MARK
302E..302F    ; Extend # Mc   [2] HANGUL SINGLE DOT TONE MARK..HANGUL DOUBLE DOT TONE MARK
3099..309A    ; Extend # Mn   [2] COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK..COMBINING KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK
A66F          ; Extend # Mn       COMBINING CYRILLIC VZMET
A670..A672    ; Extend # Me   [3] COMBINING CYRILLIC TEN MILLIONS SIGN..COMBINING CYRILLIC THOUSAND MILLIONS SIGN
A674..A67D    ; Extend # Mn  [10] COMBINING CYRILLIC LETTER UKRAINIAN IE..COMBINING CYRILLIC PAYEROK
A69E..A69F    ; Extend # Mn   [2] COMBINING CYRILLIC LETTER EF..COMBINING CYRILLIC LETTER IOTIFIED E
A6F0..A6F1    ; Extend # Mn   [2] BAMUM COMBINING MARK KOQNDON..BAMUM COMBINING MARK TUKWENTIS
A802          ; Extend # Mn       SYLOTI NAGRI SIGN DVISVARA
A806          ; Extend # Mn       SYLOTI NAGRI SIGN HASANTA
A80B          ; Extend # Mn       SYLOTI NAGRI SIGN ANUSVARA
A825..A826    ; Extend # Mn   [2] SYLOTI NAGRI VOWEL SIGN U..SYLOTI NAGRI VOWEL SIGN E
A82C          ; Extend # Mn       SYLOTI NAGRI SIGN ALTERNATE HASANTA
A8C4..A8C5    ; Extend # Mn   [2] SAURASHTRA SIGN VIRAMA..SAURASHTRA SIGN CANDRABINDU
A8E0..A8F1    ; Extend # Mn  [18] COMBINING DEVANAGARI DIGIT ZERO..COMBINING DEVANAGARI SIGN AVAGRAHA
A8FF          ; Extend # Mn       DEVANAGARI VOWEL SIGN AY
A926..A92D    ; Extend # Mn   [8] KAYAH LI VOWEL UE..KAYAH LI TONE CALYA PLOPHU
A947..A951    ; Extend # Mn  [11] REJANG VOWEL SIGN I..REJANG CONSONANT SIGN R
A980..A982    ; Extend # Mn   [3] JAVANESE SIGN PANYANGGA..JAVANESE SIGN LAYAR
A9B3          ; Extend # Mn       JAVANESE SIGN CECAK TELU
A9B6..A9B9    ; Extend # Mn   [4] JAVANESE VOWEL SIGN WULU..JAVANESE VOWEL SIGN SUKU MENDUT
A9BC..A9BD    ; Extend # Mn   [2] JAVANESE VOWEL SIGN PEPET..JAVANESE CONSONANT SIGN KERET
A9E5          ; Extend # Mn       MYANMAR SIGN SHAN SAW
AA29..AA2E    ; Extend # Mn   [6] CHAM VOWEL SIGN AA..CHAM VOWEL SIGN OE
AA31..AA32    ; Extend # Mn   [2] CHAM VOWEL SIGN AU..CHAM VOWEL SIGN UE
AA35..AA36    ; Extend # Mn   [2] CHAM CONSONANT SIGN LA..CHAM CONSONANT SIGN WA
AA43          ; Extend # Mn       CHAM CONSONANT SIGN FINAL NG
AA4C          ; Extend # Mn       CHAM CONSONANT SIGN FINAL M
AA7C          ; Extend # Mn       MYANMAR SIGN TAI LAING TONE-2
AAB0          ; Extend # Mn       TAI VIET MAI KANG
AAB2..AAB4    ; Extend # Mn   [3] TAI VIET VOWEL I..TAI VIET VOWEL U
AAB7..AAB8    ; Extend # Mn   [2] TAI VIET MAI KHIT..TAI VIET VOWEL IA
AABE..AABF    ; Extend # Mn   [2] TAI VIET VOWEL AM..TAI VIET TONE MAI EK
AAC1          ; Extend # Mn       TAI VIET TONE MAI THO
AAEC..AAED    ; Extend # Mn   [2] MEETEI MAYEK VOWEL SIGN UU..MEETEI MAYEK VOWEL SIGN AAI
AAF6          ; Extend # Mn       MEETEI MAYEK VIRAMA
ABE5          ; Extend # Mn       MEETEI MAYEK VOWEL SIGN ANAP
ABE8          ; Extend # Mn       MEETEI MAYEK VOWEL SIGN UNAP
ABED          ; Extend # Mn       MEETEI MAYEK APUN IYEK
FB1E          ; Extend # Mn       HEBREW POINT JUDEO-SPANISH VARIKA
FE00..FE0F    ; Extend # Mn  [16] VARIATION SELECTOR-1..VARIATION SELECTOR-16
FE20..FE2F    ; Extend # Mn  [16] COMBINING LIGATURE LEFT HALF..COMBINING CYRILLIC TITLO RIGHT HALF
FF9E..FF9F    ; Extend # Lm   [2] HALFWIDTH KATAKANA VOICED SOUND MARK..HALFWIDTH KATAKANA SEMI-VOICED SOUND MARK
101FD         ; Extend # Mn       PHAISTOS DISC SIGN COMBINING OBLIQUE STROKE
102E0         ; Extend # Mn       COPTIC EPACT THOUSANDS MARK
10376..1037A  ; Extend # Mn   [5] COMBINING OLD PERMIC LETTER AN..COMBINING OLD PERMIC LETTER SII
10A01..10A03  ; Extend # Mn   [3] KHAROSHTHI VOWEL SIGN I..KHAROSHTHI VOWEL SIGN VOCALIC R
10A05..10A06  ; Extend # Mn   [2] KHAROSHTHI VOWEL SIGN E..KHAROSHTHI VOWEL SIGN O
10A0C..10A0F  ; Extend # Mn   [4] KHAROSHTHI VOWEL LENGTH MARK..KHAROSHTHI SIGN VISARGA
10A38..10A3A  ; Extend # Mn   [3] KHAROSHTHI SIGN BAR ABOVE..KHAROSHTHI SIGN DOT BELOW
10A3F         ; Extend # Mn       KHAROSHTHI VIRAMA
10AE5..10AE6  ; Extend # Mn   [2] MANICHAEAN ABBREVIATION MARK ABOVE..MANICHAEAN ABBREVIATION MARK BELOW
10D24..10D27  ; Extend # Mn   [4] HANIFI ROHINGYA SIGN HARBAHAY..HANIFI ROHINGYA SIGN TASSI
10EAB..10EAC  ; Extend # Mn   [2] YEZIDI COMBINING HAMZA MARK..YEZIDI COMBINING MADDA MARK
10EFD..10EFF  ; Extend # Mn   [3] ARABIC SMALL LOW WORD SAKTA..ARABIC SMALL LOW WORD MADDA
10F46..10F50  ; Extend # Mn  [11] SOGDIAN COMBINING DOT BELOW..SOGDIAN COMBINING STROKE BELOW
10F82..10F85  ; Extend # Mn   [4] OLD UYGHUR COMBINING DOT ABOVE..OLD UYGHUR COMBINING TWO DOTS BELOW
11001         ; Extend # Mn       BRAHMI SIGN ANUSVARA
11038..11046  ; Extend # Mn  [15] BRAHMI VOWEL SIGN AA..BRAHMI VIRAMA
11070         ; Extend # Mn       BRAHMI SIGN OLD TAMIL VIRAMA
11073..11074  ; Extend # Mn   [2] BRAHMI VOWEL SIGN OLD TAMIL SHORT E..BRAHMI VOWEL SIGN OLD TAMIL SHORT O
1107F..11081  ; Extend # Mn   [3] BRAHMI NUMBER JOINER..KAITHI SIGN ANUSVARA
110B3..110B6  ; Extend # Mn   [4] KAITHI VOWEL SIGN U..KAITHI VOWEL SIGN AI
110B9..110BA  ; Extend # Mn   [2] KAITHI SIGN VIRAMA..KAITHI SIGN NUKTA
110C2         ; Extend # Mn       KAITHI VOWEL SIGN VOCALIC R
11100..11102  ; Extend # Mn   [3] CHAKMA SIGN CANDRABINDU..CHAKMA SIGN VISARGA
11127..1112B  ; Extend # Mn   [5] CHAKMA VOWEL SIGN A..CHAKMA VOWEL SIGN UU
1112D..11134  ; Extend # Mn   [8] CHAKMA VOWEL SIGN AI..CHAKMA MAAYYAA
11173         ; Extend # Mn       MAHAJANI SIGN NUKTA
11180..11181  ; Extend # Mn   [2] SHARADA SIGN CANDRABINDU..SHARADA SIGN ANUSVARA
111B6..111BE  ; Extend # Mn   [9] SHARADA VOWEL SIGN U..SHARADA VOWEL SIGN O
111C9..111CC  ; Extend # Mn   [4] SHARADA SANDHI MARK..SHARADA EXTRA SHORT VOWEL MARK
111CF         ; Extend # Mn       SHARADA SIGN INVERTED CANDRABINDU
1122F..11231  ; Extend # Mn   [3] KHOJKI VOWEL SIGN U..KHOJKI VOWEL SIGN AI
11234         ; Extend # Mn       KHOJKI SIGN ANUSVARA
11236..11237  ; Extend # Mn   [2] KHOJKI SIGN NUKTA..KHOJKI SIGN SHADDA
1123E         ; Extend # Mn       KHOJKI SIGN SUKUN
11241         ; Extend # Mn       KHOJKI VOWEL SIGN VOCALIC R
112DF         ; Extend # Mn       KHUDAWADI SIGN ANUSVARA
112E3..112EA  ; Extend # Mn   [8] KHUDAWADI VOWEL SIGN U..KHUDAWADI SIGN VIRAMA
11300..11301  ; Extend # Mn   [2] GRANTHA SIGN COMBINING ANUSVARA ABOVE..GRANTHA SIGN CANDRABINDU
1133B..1133C  ; Extend # Mn   [2] COMBINING BINDU BELOW..GRANTHA SIGN NUKTA
1133E         ; Extend # Mc       GRANTHA VOWEL SIGN AA
11340         ; Extend # Mn       GRANTHA VOWEL SIGN II
11357         ; Extend # Mc       GRANTHA AU LENGTH MARK
11366..1136C  ; Extend # Mn   [7] COMBINING GRANTHA DIGIT ZERO..COMBINING GRANTHA DIGIT SIX
11370..11374  ; Extend # Mn   [5] COMBINING GRANTHA LETTER A..COMBINING GRANTHA LETTER PA
11438..1143F  ; Extend # Mn   [8] NEWA VOWEL SIGN U..NEWA VOWEL SIGN AI
11442..11444  ; Extend # Mn   [3] NEWA SIGN VIRAMA..NEWA SIGN ANUSVARA
11446         ; Extend # Mn       NEWA SIGN NUKTA
1145E         ; Extend # Mn       NEWA SANDHI MARK
114B0         ; Extend # Mc       TIRHUTA VOWEL SIGN AA
114B3..114B8  ; Extend # Mn   [6] TIRHUTA VOWEL SIGN U..TIRHUTA VOWEL SIGN VOCALIC LL
114BA         ; Extend # Mn       TIRHUTA VOWEL SIGN SHORT E
114BD         ; Extend # Mc       TIRHUTA VOWEL SIGN SHORT O
114BF..114C0  ; Extend # Mn   [2] TIRHUTA SIGN CANDRABINDU..TIRHUTA SIGN ANUSVARA
114C2..114C3  ; Extend # Mn   [2] TIRHUTA SIGN VIRAMA..TIRHUTA SIGN NUKTA
115AF         ; Extend # Mc       SIDDHAM VOWEL SIGN AA
115B2..115B5  ; Extend # Mn   [4] SIDDHAM VOWEL SIGN U..SIDDHAM VOWEL SIGN VOCALIC RR
115BC..115BD  ; Extend # Mn   [2] SIDDHAM SIGN CANDRABINDU..SIDDHAM SIGN ANUSVARA
115BF..115C0  ; Extend # Mn   [2] SIDDHAM SIGN VIRAMA..SIDDHAM SIGN NUKTA
115DC..115DD  ; Extend # Mn   [2] SIDDHAM VOWEL SIGN ALTERNATE U..SIDDHAM VOWEL SIGN ALTERNATE UU
11633..1163A  ; Extend # Mn   [8] MODI VOWEL SIGN U..MODI VOWEL SIGN AI
1163D         ; Extend # Mn       MODI SIGN ANUSVARA
1163F..11640  ; Extend # Mn   [2] MODI SIGN VIRAMA..MODI SIGN ARDHACANDRA
116AB         ; Extend # Mn       TAKRI SIGN ANUSVARA
116AD         ; Extend # Mn       TAKRI VOWEL SIGN AA
116B0..116B5  ; Extend # Mn   [6] TAKRI VOWEL SIGN U..TAKRI VOWEL SIGN AU
116B7         ; Extend # Mn       TAKRI SIGN NUKTA
1171D..1171F  ; Extend # Mn   [3] AHOM CONSONANT SIGN MEDIAL LA..AHOM CONSONANT SIGN MEDIAL LIGATING RA
11722..11725  ; Extend # Mn   [4] AHOM VOWEL SIGN I..AHOM VOWEL SIGN UU
11727..1172B  ; Extend # Mn   [5] AHOM VOWEL SIGN AW..AHOM SIGN KILLER
1182F..11837  ; Extend # Mn   [9] DOGRA VOWEL SIGN U..DOGRA SIGN ANUSVARA
11839..1183A  ; Extend # Mn   [2] DOGRA SIGN VIRAMA..DOGRA SIGN NUKTA
11930         ; Extend # Mc       DIVES AKURU VOWEL SIGN AA
1193B..1193C  ; Extend # Mn   [2] DIVES AKURU SIGN ANUSVARA..DIVES AKURU SIGN CANDRABINDU
1193E         ; Extend # Mn       DIVES AKURU VIRAMA
11943         ; Extend # Mn       DIVES AKURU SIGN NUKTA
119D4..119D7  ; Extend # Mn   [4] NANDINAGARI VOWEL SIGN U..NANDINAGARI VOWEL SIGN VOCALIC RR
119DA..119DB  ; Extend # Mn   [2] NANDINAGARI VOWEL SIGN E..NANDINAGARI VOWEL SIGN AI
119E0         ; Extend # Mn       NANDINAGARI SIGN VIRAMA
11A01..11A0A  ; Extend # Mn  [10] ZANABAZAR SQUARE VOWEL SIGN I..ZANABAZAR SQUARE VOWEL LENGTH MARK
11A33..11A38  ; Extend # Mn   [6] ZANABAZAR SQUARE FINAL CONSONANT MARK..ZANABAZAR SQUARE SIGN ANUSVARA
11A3B..11A3E  ; Extend # Mn   [4] ZANABAZAR SQUARE CLUSTER-FINAL LETTER YA..ZANABAZAR SQUARE CLUSTER-FINAL LETTER VA
11A47         ; Extend # Mn       ZANABAZAR SQUARE SUBJOINER
11A51..11A56  ; Extend # Mn   [6] SOYOMBO VOWEL SIGN I..SOYOMBO VOWEL SIGN OE
11A59..11A5B  ; Extend # Mn   [3] SOYOMBO VOWEL SIGN VOCALIC R..SOYOMBO VOWEL LENGTH MARK
11A8A..11A96  ; Extend # Mn  [13] SOYOMBO FINAL CONSONANT SIGN G..SOYOMBO SIGN ANUSVARA
11A98..11A99  ; Extend # Mn   [2] SOYOMBO GEMINATION MARK..SOYOMBO SUBJOINER
11C30..11C36  ; Extend # Mn   [7] BHAIKSUKI VOWEL SIGN I..BHAIKSUKI VOWEL SIGN VOCALIC L
11C38..11C3D  ; Extend # Mn   [6] BHAIKSUKI VOWEL SIGN E..BHAIKSUKI SIGN ANUSVARA
11C3F         ; Extend # Mn       BHAIKSUKI SIGN VIRAMA
11C92..11CA7  ; Extend # Mn  [22] MARCHEN SUBJOINED LETTER KA..MARCHEN SUBJOINED LETTER ZA
11CAA..11CB0  ; Extend # Mn   [7] MARCHEN SUBJOINED LETTER RA..MARCHEN VOWEL SIGN AA
11CB2..11CB3  ; Extend # Mn   [2] MARCHEN VOWEL SIGN U..MARCHEN VOWEL SIGN E
11CB5..11CB6  ; Extend # Mn   [2] MARCHEN SIGN ANUSVARA..MARCHEN SIGN CANDRABINDU
11D31..11D36  ; Extend # Mn   [6] MASARAM GONDI VOWEL SIGN AA..MASARAM GONDI VOWEL SIGN VOCALIC R
11D3A         ; Extend # Mn       MASARAM GONDI VOWEL SIGN E
11D3C..11D3D  ; Extend # Mn   [2] MASARAM GONDI VOWEL SIGN AI..MASARAM GONDI VOWEL SIGN O
11D3F..11D45  ; Extend # Mn   [7] MASARAM GONDI VOWEL SIGN AU..MASARAM GONDI VIRAMA
11D47         ; Extend # Mn       MASARAM GONDI RA-KARA
11D90..11D91  ; Extend # Mn   [2] GUNJALA GONDI VOWEL SIGN EE..GUNJALA GONDI VOWEL SIGN AI
11D95         ; Extend # Mn       GUNJALA GONDI SIGN ANUSVARA
11D97         ; Extend # Mn       GUNJALA GONDI VIRAMA
11EF3..11EF4  ; Extend # Mn   [2] MAKASAR VOWEL SIGN I..MAKASAR VOWEL SIGN U
11F00..11F01  ; Extend # Mn   [2] KAWI SIGN CANDRABINDU..KAWI SIGN ANUSVARA
11F36..11F3A  ; Extend # Mn   [5] KAWI VOWEL SIGN I..KAWI VOWEL SIGN VOCALIC R
11F40         ; Extend # Mn       KAWI VOWEL SIGN EU
11F42         ; Extend # Mn       KAWI CONJOINER
13440         ; Extend # Mn       EGYPTIAN HIEROGLYPH MIRROR HORIZONTALLY
13447..13455  ; Extend # Mn  [15] EGYPTIAN HIEROGLYPH MODIFIER DAMAGED AT TOP START..EGYPTIAN HIEROGLYPH MODIFIER DAMAGED
16AF0..16AF4  ; Extend # Mn   [5] BASSA VAH COMBINING HIGH TONE..BASSA VAH COMBINING HIGH-LOW TONE
16B30..16B36  ; Extend # Mn   [7] PAHAWH HMONG MARK CIM TUB..PAHAWH HMONG MARK CIM TAUM
16F4F         ; Extend # Mn       MIAO SIGN CONSONANT MODIFIER BAR
16F8F..16F92  ; Extend # Mn   [4] MIAO TONE RIGHT..MIAO TONE BELOW
16FE4         ; Extend # Mn       KHITAN SMALL SCRIPT FILLER
1BC9D..1BC9E  ; Extend # Mn   [2] DUPLOYAN THICK LETTER SELECTOR..DUPLOYAN DOUBLE MARK
1CF00..1CF2D  ; Extend # Mn  [46] ZNAMENNY COMBINING MARK GORAZDO NIZKO S KRYZHEM ON LEFT..ZNAMENNY COMBINING MARK KRYZH ON LEFT
1CF30..1CF46  ; Extend # Mn  [23] ZNAMENNY COMBINING TONAL RANGE MARK MRACHNO..ZNAMENNY PRIZNAK MODIFIER ROG
1D165         ; Extend # Mc       MUSICAL SYMBOL COMBINING STEM
1D167..1D169  ; Extend # Mn   [3] MUSICAL SYMBOL COMBINING TREMOLO-1..MUSICAL SYMBOL COMBINING TREMOLO-3
1D16E..1D172  ; Extend # Mc   [5] MUSICAL SYMBOL COMBINING FLAG-1..MUSICAL SYMBOL COMBINING FLAG-5
1D17B..1D182  ; Extend # Mn   [8] MUSICAL SYMBOL COMBINING ACCENT..MUSICAL SYMBOL COMBINING LOURE
1D185..1D18B  ; Extend # Mn   [7] MUSICAL SYMBOL COMBINING DOIT..MUSICAL SYMBOL COMBINING TRIPLE TONGUE
1D1AA..1D1AD  ; Extend # Mn   [4] MUSICAL SYMBOL COMBINING DOWN BOW..MUSICAL SYMBOL COMBINING SNAP PIZZICATO
1D242..1D244  ; Extend # Mn   [3] COMBINING GREEK MUSICAL TRISEME..COMBINING GREEK MUSICAL PENTASEME
1DA00..1DA36  ; Extend # Mn  [55] SIGNWRITING HEAD RIM..SIGNWRITING AIR SUCKING IN
1DA3B..1DA6C  ; Extend # Mn  [50] SIGNWRITING MOUTH CLOSED NEUTRAL..SIGNWRITING EXCITEMENT
1DA75         ; Extend # Mn       SIGNWRITING UPPER BODY TILTING FROM HIP JOINTS
1DA84         ; Extend # Mn       SIGNWRITING LOCATION HEAD NECK
1DA9B..1DA9F  ; Extend # Mn   [5] SIGNWRITING FILL MODIFIER-2..SIGNWRITING FILL MODIFIER-6
1DAA1..1DAAF  ; Extend # Mn  [15] SIGNWRITING ROTATION MODIFIER-2..SIGNWRITING ROTATION MODIFIER-16
1E000..1E006  ; Extend # Mn   [7] COMBINING GLAGOLITIC LETTER AZU..COMBINING GLAGOLITIC LETTER ZHIVETE
1E008..1E018  ; Extend # Mn  [17] COMBINING GLAGOLITIC LETTER ZEMLJA..COMBINING GLAGOLITIC LETTER HERU
1E01B..1E021  ; Extend # Mn   [7] COMBINING GLAGOLITIC LETTER SHTA..COMBINING GLAGOLITIC LETTER YATI
1E023..1E024  ; Extend # Mn   [2] COMBINING GLAGOLITIC LETTER YU..COMBINING GLAGOLITIC LETTER SMALL YUS
1E026..1E02A  ; Extend # Mn   [5] COMBINING GLAGOLITIC LETTER YO..COMBINING GLAGOLITIC LETTER FITA
1E08F         ; Extend # Mn       COMBINING CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
1E130..1E136  ; Extend # Mn   [7] NYIAKENG PUACHUE HMONG TONE-B..NYIAKENG PUACHUE HMONG TONE-D
1E2AE         ; Extend # Mn       TOTO SIGN RISING TONE
1E2EC..1E2EF  ; Extend # Mn   [4] WANCHO TONE TUP..WANCHO TONE KOINI
1E4EC..1E4EF  ; Extend # Mn   [4] NAG MUNDARI SIGN MUHOR..NAG MUNDARI SIGN SUTUH
1E8D0..1E8D6  ; Extend # Mn   [7] MENDE KIKAKUI COMBINING NUMBER TEENS..MENDE KIKAKUI COMBINING NUMBER MILLIONS
1E944..1E94A  ; Extend # Mn   [7] ADLAM ALIF LENGTHENER..ADLAM NUKTA
1F3FB..1F3FF  ; Extend # Sk   [5] EMOJI MODIFIER FITZPATRICK TYPE-1-2..EMOJI MODIFIER FITZPATRICK TYPE-6
E0020..E007F  ; Extend # Cf  [96] TAG SPACE..CANCEL TAG
E0100..E01EF  ; Extend # Mn [240] VARIATION SELECTOR-17..VARIATION SELECTOR-256

# Total code points: 2130

# ================================================

1F1E6..1F1FF  ; Regional_Indicator # So  [26] REGIONAL INDICATOR SYMBOL LETTER A..REGIONAL INDICATOR SYMBOL LETTER Z

# Total code points: 26

# ================================================

0903          ; SpacingMark # Mc       DEVANAGARI SIGN VISARGA
093B          ; SpacingMark # Mc       DEVANAGARI VOWEL SIGN OOE
093E..0940    ; SpacingMark # Mc   [3] DEVANAGARI VOWEL SIGN AA..DEVANAGARI VOWEL SIGN II
0949..094C    ; SpacingMark # Mc   [4] DEVANAGARI VOWEL SIGN CANDRA O..DEVANAGARI VOWEL SIGN AU
094E..094F    ; SpacingMark # Mc   [2] DEVANAGARI VOWEL SIGN PRISHTHAMATRA E..DEVANAGARI VOWEL SIGN AW
0982..0983    ; SpacingMark # Mc   [2] BENGALI SIGN ANUSVARA..BENGALI SIGN VISARGA
09BF..09C0    ; SpacingMark # Mc   [2] BENGALI VOWEL SIGN I..BENGALI VOWEL SIGN II
09C7..09C8    ; SpacingMark # Mc   [2] BENGALI VOWEL SIGN E..BENGALI VOWEL SIGN AI
09CB..09CC    ; SpacingMark # Mc   [2] BENGALI VOWEL SIGN O..BENGALI VOWEL SIGN AU
0A03          ; SpacingMark # Mc       GURMUKHI SIGN VISARGA
0A3E..0A40    ; SpacingMark # Mc   [3] GURMUKHI VOWEL SIGN AA..GURMUKHI VOWEL SIGN II
0A83          ; SpacingMark # Mc       GUJARATI SIGN VISARGA
0ABE..0AC0    ; SpacingMark # Mc   [3] GUJARATI VOWEL SIGN AA..GUJARATI VOWEL SIGN II
0AC9          ; SpacingMark # Mc       GUJARATI VOWEL SIGN CANDRA O
0ACB..0ACC    ; SpacingMark # Mc   [2] GUJARATI VOWEL SIGN O..GUJARATI VOWEL SIGN AU
0B02..0B03    ; SpacingMark # Mc   [2] ORIYA SIGN ANUSVARA..ORIYA SIGN VISARGA
0B40          ; SpacingMark # Mc       ORIYA VOWEL SIGN II
0B47..0B48    ; SpacingMark # Mc   [2] ORIYA VOWEL SIGN E..ORIYA VOWEL SIGN AI
0B4B..0B4C    ; SpacingMark # Mc   [2] ORIYA VOWEL SIGN O..ORIYA VOWEL SIGN AU
0BBF          ; SpacingMark # Mc       TAMIL VOWEL SIGN I
0BC1..0BC2    ; SpacingMark # Mc   [2] TAMIL VOWEL SIGN U..TAMIL VOWEL SIGN UU
0BC6..0BC8    ; SpacingMark # Mc   [3] TAMIL VOWEL SIGN E..TAMIL VOWEL SIGN AI
0BCA..0BCC    ; SpacingMark # Mc   [3] TAMIL VOWEL SIGN O..TAMIL VOWEL SIGN AU
0C01..0C03    ; SpacingMark # Mc   [3] TELUGU SIGN CANDRABINDU..TELUGU SIGN VISARGA
0C41..0C44    ; SpacingMark # Mc   [4] TELUGU VOWEL SIGN U..TELUGU VOWEL SIGN VOCALIC RR
0C82..0C83    ; SpacingMark # Mc   [2] KANNADA SIGN ANUSVARA..KANNADA SIGN VISARGA
0CBE          ; SpacingMark # Mc       KANNADA VOWEL SIGN AA
0CC0..0CC1    ; SpacingMark # Mc   [2] KANNADA VOWEL SIGN II..KANNADA VOWEL SIGN U
0CC3..0CC4    ; SpacingMark # Mc   [2] KANNADA VOWEL SIGN VOCALIC R..KANNADA VOWEL SIGN VOCALIC RR
0CC7..0CC8    ; SpacingMark # Mc   [2] KANNADA VOWEL SIGN EE..KANNADA VOWEL SIGN AI
0CCA..0CCB    ; SpacingMark # Mc   [2] KANNADA VOWEL SIGN O..KANNADA VOWEL SIGN OO
0CF3          ; SpacingMark # Mc       KANNADA SIGN COMBINING ANUSVARA ABOVE RIGHT
0D02..0D03    ; SpacingMark # Mc   [2] MALAYALAM SIGN ANUSVARA..MALAYALAM SIGN VISARGA
0D3F..0D40    ; SpacingMark # Mc   [2] MALAYALAM VOWEL SIGN I..MALAYALAM VOWEL SIGN II
0D46..0D48    ; SpacingMark # Mc   [3] MALAYALAM VOWEL SIGN E..MALAYALAM VOWEL SIGN AI
0D4A..0D4C    ; SpacingMark # Mc   [3] MALAYALAM VOWEL SIGN O..MALAYALAM VOWEL SIGN AU
0D82..0D83    ; SpacingMark # Mc   [2] SINHALA SIGN ANUSVARAYA..SINHALA SIGN VISARGAYA
0DD0..0DD1    ; SpacingMark # Mc   [2] SINHALA VOWEL SIGN KETTI AEDA-PILLA..SINHALA VOWEL SIGN DIGA AEDA-PILLA
0DD8..0DDE    ; SpacingMark # Mc   [7] SINHALA VOWEL SIGN GAETTA-PILLA..SINHALA VOWEL SIGN KOMBUVA HAA GAYANUKITTA
0DF2..0DF3    ; SpacingMark # Mc   [2] SINHALA VOWEL SIGN DIGA GAETTA-PILLA..SINHALA VOWEL SIGN DIGA GAYANUKITTA
0E33          ; SpacingMark # Lo       THAI CHARACTER SARA AM
0EB3          ; SpacingMark # Lo       LAO VOWEL SIGN AM
0F3E..0F3F    ; SpacingMark # Mc   [2] TIBETAN SIGN YAR TSHES..TIBETAN SIGN MAR TSHES
0F7F          ; SpacingMark # Mc       TIBETAN SIGN RNAM BCAD
1031          ; SpacingMark # Mc       MYANMAR VOWEL SIGN E
103B..103C    ; SpacingMark # Mc   [2] MYANMAR CONSONANT SIGN MEDIAL YA..MYANMAR CONSONANT SIGN MEDIAL RA
1056..1057    ; SpacingMark # Mc   [2] MYANMAR VOWEL SIGN VOCALIC R..MYANMAR VOWEL SIGN VOCALIC RR
1084          ; SpacingMark # Mc       MYANMAR VOWEL SIGN SHAN E
1715          ; SpacingMark # Mc       TAGALOG SIGN PAMUDPOD
1734          ; SpacingMark # Mc       HANUNOO SIGN PAMUDPOD
17B6          ; SpacingMark # Mc       KHMER VOWEL SIGN AA
17BE..17C5    ; SpacingMark # Mc   [8] KHMER VOWEL SIGN OE..KHMER VOWEL SIGN AU
17C7..17C8    ; SpacingMark # Mc   [2] KHMER SIGN REAHMUK..KHMER SIGN YUUKALEAPINTU
1923..1926    ; SpacingMark # Mc   [4] LIMBU VOWEL SIGN EE..LIMBU VOWEL SIGN AU
1929..192B    ; SpacingMark # Mc   [3] LIMBU SUBJOINED LETTER YA..LIMBU SUBJOINED LETTER WA
1930..1931    ; SpacingMark # Mc   [2] LIMBU SMALL LETTER KA..LIMBU SMALL LETTER NGA
1933..1938    ; SpacingMark # Mc   [6] LIMBU SMALL LETTER TA..LIMBU SMALL LETTER LA
1A19..1A1A    ; SpacingMark # Mc   [2] BUGINESE VOWEL SIGN E..BUGINESE VOWEL SIGN O
1A55          ; SpacingMark # Mc       TAI THAM CONSONANT SIGN MEDIAL RA
1A57          ; SpacingMark # Mc       TAI THAM CONSONANT SIGN LA TANG LAI
1A6D..1A72    ; SpacingMark # Mc   [6] TAI THAM VOWEL SIGN OY..TAI THAM VOWEL SIGN THAM AI
1B04          ; SpacingMark # Mc       BALINESE SIGN BISAH
1B3B          ; SpacingMark # Mc       BALINESE VOWEL SIGN RA REPA TEDUNG
1B3D..1B41    ; SpacingMark # Mc   [5] BALINESE VOWEL SIGN LA LENGA TEDUNG..BALINESE VOWEL SIGN TALING REPA TEDUNG
1B43..1B44    ; SpacingMark # Mc   [2] BALINESE VOWEL SIGN PEPET TEDUNG..BALINESE ADEG ADEG
1B82          ; SpacingMark # Mc       SUNDANESE SIGN PANGWISAD
1BA1          ; SpacingMark # Mc       SUNDANESE CONSONANT SIGN PAMINGKAL
1BA6..1BA7    ; SpacingMark # Mc   [2] SUNDANESE VOWEL SIGN PANAELAENG..SUNDANESE VOWEL SIGN PANOLONG
1BAA          ; SpacingMark # Mc       SUNDANESE SIGN PAMAAEH
1BE7          ; SpacingMark # Mc       BATAK VOWEL SIGN E
1BEA..1BEC    ; SpacingMark # Mc   [3] BATAK VOWEL SIGN I..BATAK VOWEL SIGN O
1BEE          ; SpacingMark # Mc       BATAK VOWEL SIGN U
1BF2..1BF3    ; SpacingMark # Mc   [2] BATAK PANGOLAT..BATAK PANONGONAN
1C24..1C2B    ; SpacingMark # Mc   [8] LEPCHA SUBJOINED LETTER YA..LEPCHA VOWEL SIGN UU
1C34..1C35    ; SpacingMark # Mc   [2] LEPCHA CONSONANT SIGN NYIN-DO..LEPCHA CONSONANT SIGN KANG
1CE1          ; SpacingMark # Mc       VEDIC TONE ATHARVAVEDIC INDEPENDENT SVARITA
1CF7          ; SpacingMark # Mc       VEDIC SIGN ATIKRAMA
A823..A824    ; SpacingMark # Mc   [2] SYLOTI NAGRI VOWEL SIGN A..SYLOTI NAGRI VOWEL SIGN I
A827          ; SpacingMark # Mc       SYLOTI NAGRI VOWEL SIGN OO
A880..A881    ; SpacingMark # Mc   [2] SAURASHTRA SIGN ANUSVARA..SAURASHTRA SIGN VISARGA
A8B4..A8C3    ; SpacingMark # Mc  [16] SAURASHTRA CONSONANT SIGN HAARU..SAURASHTRA VOWEL SIGN AU
A952..A953    ; SpacingMark # Mc   [2] REJANG CONSONANT SIGN H..REJANG VIRAMA
A983          ; SpacingMark # Mc       JAVANESE SIGN WIGNYAN
A9B4..A9B5    ; SpacingMark # Mc   [2] JAVANESE VOWEL SIGN TARUNG..JAVANESE VOWEL SIGN TOLONG
A9BA..A9BB    ; SpacingMark # Mc   [2] JAVANESE VOWEL SIGN TALING..JAVANESE VOWEL SIGN DIRGA MURE
A9BE..A9C0    ; SpacingMark # Mc   [3] JAVANESE CONSONANT SIGN PENGKAL..JAVANESE PANGKON
AA2F..AA30    ; SpacingMark # Mc   [2] CHAM VOWEL SIGN O..CHAM VOWEL SIGN AI
AA33..AA34    ; SpacingMark # Mc   [2] CHAM CONSONANT SIGN YA..CHAM CONSONANT SIGN RA
AA4D          ; SpacingMark # Mc       CHAM CONSONANT SIGN FINAL H
AAEB          ; SpacingMark # Mc       MEETEI MAYEK VOWEL SIGN II
AAEE..AAEF    ; SpacingMark # Mc   [2] MEETEI MAYEK VOWEL SIGN AU..MEETEI MAYEK VOWEL SIGN AAU
AAF5          ; SpacingMark # Mc       MEETEI MAYEK VOWEL SIGN VISARGA
ABE3..ABE4    ; SpacingMark # Mc   [2] MEETEI MAYEK VOWEL SIGN ONAP..MEETEI MAYEK VOWEL SIGN INAP
ABE6..ABE7    ; SpacingMark # Mc   [2] MEETEI MAYEK VOWEL SIGN YENAP..MEETEI MAYEK VOWEL SIGN SOUNAP
ABE9..ABEA    ; SpacingMark # Mc   [2] MEETEI MAYEK VOWEL SIGN CHEINAP..MEETEI MAYEK VOWEL SIGN NUNG
ABEC          ; SpacingMark # Mc       MEETEI MAYEK LUM IYEK
11000         ; SpacingMark # Mc       BRAHMI SIGN CANDRABINDU
11002         ; SpacingMark # Mc       BRAHMI SIGN VISARGA
11082         ; SpacingMark # Mc       KAITHI SIGN VISARGA
110B0..110B2  ; SpacingMark # Mc   [3] KAITHI VOWEL SIGN AA..KAITHI VOWEL SIGN II
110B7..110B8  ; SpacingMark # Mc   [2] KAITHI VOWEL SIGN O..KAITHI VOWEL SIGN AU
1112C         ; SpacingMark # Mc       CHAKMA VOWEL SIGN E
11145..11146  ; SpacingMark # Mc   [2] CHAKMA VOWEL SIGN AA..CHAKMA VOWEL SIGN EI
11182         ; SpacingMark # Mc       SHARADA SIGN VISARGA
111B3..111B5  ; SpacingMark # Mc   [3] SHARADA VOWEL SIGN AA..SHARADA VOWEL SIGN II
111BF..111C0  ; SpacingMark # Mc   [2] SHARADA VOWEL SIGN AU..SHARADA SIGN VIRAMA
111CE         ; SpacingMark # Mc       SHARADA VOWEL SIGN PRISHTHAMATRA E
1122C..1122E  ; SpacingMark # Mc   [3] KHOJKI VOWEL SIGN AA..KHOJKI VOWEL SIGN II
11232..11233  ; SpacingMark # Mc   [2] KHOJKI VOWEL SIGN O..KHOJKI VOWEL SIGN AU
11235         ; SpacingMark # Mc       KHOJKI SIGN VIRAMA
112E0..112E2  ; SpacingMark # Mc   [3] KHUDAWADI VOWEL SIGN AA..KHUDAWADI VOWEL SIGN II
11302..11303  ; SpacingMark # Mc   [2] GRANTHA SIGN ANUSVARA..GRANTHA SIGN VISARGA
1133F         ; SpacingMark # Mc       GRANTHA VOWEL SIGN I
11341..11344  ; SpacingMark # Mc   [4] GRANTHA VOWEL SIGN U..GRANTHA VOWEL SIGN VOCALIC RR
11347..11348  ; SpacingMark # Mc   [2] GRANTHA VOWEL SIGN EE..GRANTHA VOWEL SIGN AI
1134B..1134D  ; SpacingMark # Mc   [3] GRANTHA VOWEL SIGN OO..GRANTHA SIGN VIRAMA
11362..11363  ; SpacingMark # Mc   [2] GRANTHA VOWEL SIGN VOCALIC L..GRANTHA VOWEL SIGN VOCALIC LL
11435..11437  ; SpacingMark # Mc   [3] NEWA VOWEL SIGN AA..NEWA VOWEL SIGN II
11440..11441  ; SpacingMark # Mc   [2] NEWA VOWEL SIGN O..NEWA VOWEL SIGN AU
11445         ; SpacingMark # Mc       NEWA SIGN VISARGA
114B1..114B2  ; SpacingMark # Mc   [2] TIRHUTA VOWEL SIGN I..TIRHUTA VOWEL SIGN II
114B9         ; SpacingMark # Mc       TIRHUTA VOWEL SIGN E
114BB..114BC  ; SpacingMark # Mc   [2] TIRHUTA VOWEL SIGN AI..TIRHUTA VOWEL SIGN O
114BE         ; SpacingMark # Mc       TIRHUTA VOWEL SIGN AU
114C1         ; SpacingMark # Mc       TIRHUTA SIGN VISARGA
115B0..115B1  ; SpacingMark # Mc   [2] SIDDHAM VOWEL SIGN I..SIDDHAM VOWEL SIGN II
115B8..115BB  ; SpacingMark # Mc   [4] SIDDHAM VOWEL SIGN E..SIDDHAM VOWEL SIGN AU
115BE         ; SpacingMark # Mc       SIDDHAM SIGN VISARGA
11630..11632  ; SpacingMark # Mc   [3] MODI VOWEL SIGN AA..MODI VOWEL SIGN II
1163B..1163C  ; SpacingMark # Mc   [2] MODI VOWEL SIGN O..MODI VOWEL SIGN AU
1163E         ; SpacingMark # Mc       MODI SIGN VISARGA
116AC         ; SpacingMark # Mc       TAKRI SIGN VISARGA
116AE..116AF  ; SpacingMark # Mc   [2] TAKRI VOWEL SIGN I..TAKRI VOWEL SIGN II
116B6         ; SpacingMark # Mc       TAKRI SIGN VIRAMA
11726         ; SpacingMark # Mc       AHOM VOWEL SIGN E
1182C..1182E  ; SpacingMark # Mc   [3] DOGRA VOWEL SIGN AA..DOGRA VOWEL SIGN II
11838         ; SpacingMark # Mc       DOGRA SIGN VISARGA
11931..11935  ; SpacingMark # Mc   [5] DIVES AKURU VOWEL SIGN I..DIVES AKURU VOWEL SIGN E
11937..11938  ; SpacingMark # Mc   [2] DIVES AKURU VOWEL SIGN AI..DIVES AKURU VOWEL SIGN O
1193D         ; SpacingMark # Mc       DIVES AKURU SIGN HALANTA
11940         ; SpacingMark # Mc       DIVES AKURU MEDIAL YA
11942         ; SpacingMark # Mc       DIVES AKURU MEDIAL RA
119D1..119D3  ; SpacingMark # Mc   [3] NANDINAGARI VOWEL SIGN AA..NANDINAGARI VOWEL SIGN II
119DC..119DF  ; SpacingMark # Mc   [4] NANDINAGARI VOWEL SIGN O..NANDINAGARI SIGN VISARGA
119E4         ; SpacingMark # Mc       NANDINAGARI VOWEL SIGN PRISHTHAMATRA E
11A39         ; SpacingMark # Mc       ZANABAZAR SQUARE SIGN VISARGA
11A57..11A58  ; SpacingMark # Mc   [2] SOYOMBO VOWEL SIGN AI..SOYOMBO VOWEL SIGN AU
11A97         ; SpacingMark # Mc       SOYOMBO SIGN VISARGA
11C2F         ; SpacingMark # Mc       BHAIKSUKI VOWEL SIGN AA
11C3E         ; SpacingMark # Mc       BHAIKSUKI SIGN VISARGA
11CA9         ; SpacingMark # Mc       MARCHEN SUBJOINED LETTER YA
11CB1         ; SpacingMark # Mc       MARCHEN VOWEL SIGN I
11CB4         ; SpacingMark # Mc       MARCHEN VOWEL SIGN O
11D8A..11D8E  ; SpacingMark # Mc   [5] GUNJALA GONDI VOWEL SIGN AA..GUNJALA GONDI VOWEL SIGN UU
11D93..11D94  ; SpacingMark # Mc   [2] GUNJALA GONDI VOWEL SIGN OO..GUNJALA GONDI VOWEL SIGN AU
11D96         ; SpacingMark # Mc       GUNJALA GONDI SIGN VISARGA
11EF5..11EF6  ; SpacingMark # Mc   [2] MAKASAR VOWEL SIGN E..MAKASAR VOWEL SIGN O
11F03         ; SpacingMark # Mc       KAWI SIGN VISARGA
11F34..11F35  ; SpacingMark # Mc   [2] KAWI VOWEL SIGN AA..KAWI VOWEL SIGN ALTERNATE AA
11F3E..11F3F  ; SpacingMark # Mc   [2] KAWI VOWEL SIGN E..KAWI VOWEL SIGN AI
11F41         ; SpacingMark # Mc       KAWI SIGN KILLER
16F51..16F87  ; SpacingMark # Mc  [55] MIAO SIGN ASPIRATION..MIAO VOWEL SIGN UI
16FF0..16FF1  ; SpacingMark # Mc   [2] VIETNAMESE ALTERNATE READING MARK CA..VIETNAMESE ALTERNATE READING MARK NHAY
1D166         ; SpacingMark # Mc       MUSICAL SYMBOL COMBINING SPRECHGESANG STEM
1D16D         ; SpacingMark # Mc       MUSICAL SYMBOL COMBINING AUGMENTATION DOT

# Total code points: 395

# ================================================

1100..115F    ; L # Lo  [96] HANGUL CHOSEONG KIYEOK..HANGUL CHOSEONG FILLER
A960..A97C    ; L # Lo  [29] HANGUL CHOSEONG TIKEUT-MIEUM..HANGUL CHOSEONG SSANGYEORINHIEUH

# Total code points: 125

# ================================================

1160..11A7    ; V # Lo  [72] HANGUL JUNGSEONG FILLER..HANGUL JUNGSEONG O-YAE
D7B0..D7C6    ; V # Lo  [23] HANGUL JUNGSEONG O-YEO..HANGUL JUNGSEONG ARAEA-E

# Total code points: 95

# ================================================

11A8..11FF    ; T # Lo  [88] HANGUL JONGSEONG KIYEOK..HANGUL JONGSEONG SSANGNIEUN
D7CB..D7FB    ; T # Lo  [49] HANGUL JONGSEONG NIEUN-RIEUL..HANGUL JONGSEONG PHIEUPH-THIEUTH

# Total code points: 137

# ================================================

AC00          ; LV # Lo       HANGUL SYLLABLE GA
AC1C          ; LV # Lo       HANGUL SYLLABLE GAE
AC38          ; LV # Lo       HANGUL SYLLABLE GYA
AC54          ; LV # Lo       HANGUL SYLLABLE GYAE
AC70          ; LV # Lo       HANGUL SYLLABLE GEO
AC8C          ; LV # Lo       HANGUL SYLLABLE GE
ACA8          ; LV # Lo       HANGUL SYLLABLE GYEO
ACC4          ; LV # Lo       HANGUL SYLLABLE GYE
ACE0          ; LV # Lo       HANGUL SYLLABLE GO
ACFC          ; LV # Lo       HANGUL SYLLABLE GWA
AD18          ; LV # Lo       HANGUL SYLLABLE GWAE
AD34          ; LV # Lo       HANGUL SYLLABLE GOE
AD50          ; LV # Lo       HANGUL SYLLABLE GYO
AD6C          ; LV # Lo       HANGUL SYLLABLE GU
AD88          ; LV # Lo       HANGUL SYLLABLE GWEO
ADA4          ; LV # Lo       HANGUL SYLLABLE GWE
ADC0          ; LV # Lo       HANGUL SYLLABLE GWI
ADDC          ; LV # Lo       HANGUL SYLLABLE GYU
ADF8          ; LV # Lo       HANGUL SYLLABLE GEU
AE14          ; LV # Lo       HANGUL SYLLABLE GYI
AE30          ; LV # Lo       HANGUL SYLLABLE GI
AE4C          ; LV # Lo       HANGUL SYLLABLE GGA
AE68          ; LV # Lo       HANGUL SYLLABLE GGAE
AE84          ; LV # Lo       HANGUL SYLLABLE GGYA
AEA0          ; LV # Lo       HANGUL SYLLABLE GGYAE
AEBC          ; LV # Lo       HANGUL SYLLABLE GGEO
AED8          ; LV # Lo       HANGUL SYLLABLE GGE
AEF4          ; LV # Lo       HANGUL SYLLABLE GGYEO
AF10          ; LV # Lo       HANGUL SYLLABLE GGYE
AF2C          ; LV # Lo       HANGUL SYLLABLE GGO
AF48          ; LV # Lo       HANGUL SYLLABLE GGWA
AF64          ; LV # Lo       HANGUL SYLLABLE GGWAE
AF80          ; LV # Lo       HANGUL SYLLABLE GGOE
AF9C          ; LV # Lo       HANGUL SYLLABLE GGYO
AFB8          ; LV # Lo       HANGUL SYLLABLE GGU
AFD4          ; LV # Lo       HANGUL SYLLABLE GGWEO
AFF0          ; LV # Lo       HANGUL SYLLABLE GGWE
B00C          ; LV # Lo       HANGUL SYLLABLE GGWI
B028          ; LV # Lo       HANGUL SYLLABLE GGYU
B044          ; LV # Lo       HANGUL SYLLABLE GGEU
B060          ; LV # Lo       HANGUL SYLLABLE GGYI
B07C          ; LV # Lo       HANGUL SYLLABLE GGI
B098          ; LV # Lo       HANGUL SYLLABLE NA
B0B4          ; LV # Lo       HANGUL SYLLABLE NAE
B0D0          ; LV # Lo       HANGUL SYLLABLE NYA
B0EC          ; LV # Lo       HANGUL SYLLABLE NYAE
B108          ; LV # Lo       HANGUL SYLLABLE NEO
B124          ; LV # Lo       HANGUL SYLLABLE NE
B140          ; LV # Lo       HANGUL SYLLABLE NYEO
B15C          ; LV # Lo       HANGUL SYLLABLE NYE
B178          ; LV # Lo       HANGUL SYLLABLE NO
B194          ; LV # Lo       HANGUL SYLLABLE NWA
B1B0          ; LV # Lo       HANGUL SYLLABLE NWAE
B1CC          ; LV # Lo       HANGUL SYLLABLE NOE
B1E8          ; LV # Lo       HANGUL SYLLABLE NYO
B204          ; LV # Lo       HANGUL SYLLABLE NU
B220          ; LV # Lo       HANGUL SYLLABLE NWEO
B23C          ; LV # Lo       HANGUL SYLLABLE NWE
B258          ; LV # Lo       HANGUL SYLLABLE NWI
B274          ; LV # Lo       HANGUL SYLLABLE NYU
B290          ; LV # Lo       HANGUL SYLLABLE NEU
B2AC          ; LV # Lo       HANGUL SYLLABLE NYI
B2C8          ; LV # Lo       HANGUL SYLLABLE NI
B2E4          ; LV # Lo       HANGUL SYLLABLE DA
B300          ; LV # Lo       HANGUL SYLLABLE DAE
B31C          ; LV # Lo       HANGUL SYLLABLE DYA
B338          ; LV # Lo       HANGUL SYLLABLE DYAE
B354          ; LV # Lo       HANGUL SYLLABLE DEO
B370          ; LV # Lo       HANGUL SYLLABLE DE
B38C          ; LV # Lo       HANGUL SYLLABLE DYEO
B3A8          ; LV # Lo       HANGUL SYLLABLE DYE
B3C4          ; LV # Lo       HANGUL SYLLABLE DO
B3E0          ; LV # Lo       HANGUL SYLLABLE DWA
B3FC          ; LV # Lo       HANGUL SYLLABLE DWAE
B418          ; LV # Lo       HANGUL SYLLABLE DOE
B434          ; LV # Lo       HANGUL SYLLABLE DYO
B450          ; LV # Lo       HANGUL SYLLABLE DU
B46C          ; LV # Lo       HANGUL SYLLABLE DWEO
B488          ; LV # Lo       HANGUL SYLLABLE DWE
B4A4          ; LV # Lo       HANGUL SYLLABLE DWI
B4C0          ; LV # Lo       HANGUL SYLLABLE DYU
B4DC          ; LV # Lo       HANGUL SYLLABLE DEU
B4F8          ; LV # Lo       HANGUL SYLLABLE DYI
B514          ; LV # Lo       HANGUL SYLLABLE DI
B530          ; LV # Lo       HANGUL SYLLABLE DDA
B54C          ; LV # Lo       HANGUL SYLLABLE DDAE
B568          ; LV # Lo       HANGUL SYLLABLE DDYA
B584          ; LV # Lo       HANGUL SYLLABLE DDYAE
B5A0          ; LV # Lo       HANGUL SYLLABLE DDEO
B5BC          ; LV # Lo       HANGUL SYLLABLE DDE
B5D8          ; LV # Lo       HANGUL SYLLABLE DDYEO
B5F4          ; LV # Lo       HANGUL SYLLABLE DDYE
B610          ; LV # Lo       HANGUL SYLLABLE DDO
B62C          ; LV # Lo       HANGUL SYLLABLE DDWA
B648          ; LV # Lo       HANGUL SYLLABLE DDWAE
B664          ; LV # Lo       HANGUL SYLLABLE DDOE
B680          ; LV # Lo       HANGUL SYLLABLE DDYO
B69C          ; LV # Lo       HANGUL SYLLABLE DDU
B6B8          ; LV # Lo       HANGUL SYLLABLE DDWEO
B6D4          ; LV # Lo       HANGUL SYLLABLE DDWE
B6F0          ; LV # Lo       HANGUL SYLLABLE DDWI
B70C          ; LV # Lo       HANGUL SYLLABLE DDYU
B728          ; LV # Lo       HANGUL SYLLABLE DDEU
B744          ; LV # Lo       HANGUL SYLLABLE DDYI
B760          ; LV # Lo       HANGUL SYLLABLE DDI
B77C          ; LV # Lo       HANGUL SYLLABLE RA
B798          ; LV # Lo       HANGUL SYLLABLE RAE
B7B4          ; LV # Lo       HANGUL SYLLABLE RYA
B7D0          ; LV # Lo       HANGUL SYLLABLE RYAE
B7EC          ; LV # Lo       HANGUL SYLLABLE REO
B808          ; LV # Lo       HANGUL SYLLABLE RE
B824          ; LV # Lo       HANGUL SYLLABLE RYEO
B840          ; LV # Lo       HANGUL SYLLABLE RYE
B85C          ; LV # Lo       HANGUL SYLLABLE RO
B878          ; LV # Lo       HANGUL SYLLABLE RWA
B894          ; LV # Lo       HANGUL SYLLABLE RWAE
B8B0          ; LV # Lo       HANGUL SYLLABLE ROE
B8CC          ; LV # Lo       HANGUL SYLLABLE RYO
B8E8          ; LV # Lo       HANGUL SYLLABLE RU
B904          ; LV # Lo       HANGUL SYLLABLE RWEO
B920          ; LV # Lo       HANGUL SYLLABLE RWE
B93C          ; LV # Lo       HANGUL SYLLABLE RWI
B958          ; LV # Lo       HANGUL SYLLABLE RYU
B974          ; LV # Lo       HANGUL SYLLABLE REU
B990          ; LV # Lo       HANGUL SYLLABLE RYI
B9AC          ; LV # Lo       HANGUL SYLLABLE RI
B9C8          ; LV # Lo       HANGUL SYLLABLE MA
B9E4          ; LV # Lo       HANGUL SYLLABLE MAE
BA00          ; LV # Lo       HANGUL SYLLABLE MYA
BA1C          ; LV # Lo       HANGUL SYLLABLE MYAE
BA38          ; LV # Lo       HANGUL SYLLABLE MEO
BA54          ; LV # Lo       HANGUL SYLLABLE ME
BA70          ; LV # Lo       HANGUL SYLLABLE MYEO
BA8C          ; LV # Lo       HANGUL SYLLABLE MYE
BAA8          ; LV # Lo       HANGUL SYLLABLE MO
BAC4          ; LV # Lo       HANGUL SYLLABLE MWA
BAE0          ; LV # Lo       HANGUL SYLLABLE MWAE
BAFC          ; LV # Lo       HANGUL SYLLABLE MOE
BB18          ; LV # Lo       HANGUL SYLLABLE MYO
BB34          ; LV # Lo       HANGUL SYLLABLE MU
BB50          ; LV # Lo       HANGUL SYLLABLE MWEO
BB6C          ; LV # Lo       HANGUL SYLLABLE MWE
BB88          ; LV # Lo       HANGUL SYLLABLE MWI
BBA4          ; LV # Lo       HANGUL SYLLABLE MYU
BBC0          ; LV # Lo       HANGUL SYLLABLE MEU
BBDC          ; LV # Lo       HANGUL SYLLABLE MYI
BBF8          ; LV # Lo       HANGUL SYLLABLE MI
BC14          ; LV # Lo       HANGUL SYLLABLE BA
BC30          ; LV # Lo       HANGUL SYLLABLE BAE
BC4C          ; LV # Lo       HANGUL SYLLABLE BYA
BC68          ; LV # Lo       HANGUL SYLLABLE BYAE
BC84          ; LV # Lo       HANGUL SYLLABLE BEO
BCA0          ; LV # Lo       HANGUL SYLLABLE BE
BCBC          ; LV # Lo       HANGUL SYLLABLE BYEO
BCD8          ; LV # Lo       HANGUL SYLLABLE BYE
BCF4          ; LV # Lo       HANGUL SYLLABLE BO
BD10          ; LV # Lo       HANGUL SYLLABLE BWA
BD2C          ; LV # Lo       HANGUL SYLLABLE BWAE
BD48          ; LV # Lo       HANGUL SYLLABLE BOE
BD64          ; LV # Lo       HANGUL SYLLABLE BYO
BD80          ; LV # Lo       HANGUL SYLLABLE BU
BD9C          ; LV # Lo       HANGUL SYLLABLE BWEO
BDB8          ; LV # Lo       HANGUL SYLLABLE BWE
BDD4          ; LV # Lo       HANGUL SYLLABLE BWI
BDF0          ; LV # Lo       HANGUL SYLLABLE BYU
BE0C          ; LV # Lo       HANGUL SYLLABLE BEU
BE28          ; LV # Lo       HANGUL SYLLABLE BYI
BE44          ; LV # Lo       HANGUL SYLLABLE BI
BE60          ; LV # Lo       HANGUL SYLLABLE BBA
BE7C          ; LV # Lo       HANGUL SYLLABLE BBAE
BE98          ; LV # Lo       HANGUL SYLLABLE BBYA
BEB4          ; LV # Lo       HANGUL SYLLABLE BBYAE
BED0          ; LV # Lo       HANGUL SYLLABLE BBEO
BEEC          ; LV # Lo       HANGUL SYLLABLE BBE
BF08          ; LV # Lo       HANGUL SYLLABLE BBYEO
BF24          ; LV # Lo       HANGUL SYLLABLE BBYE
BF40          ; LV # Lo       HANGUL SYLLABLE BBO
BF5C          ; LV # Lo       HANGUL SYLLABLE BBWA
BF78          ; LV # Lo       HANGUL SYLLABLE BBWAE
BF94          ; LV # Lo       HANGUL SYLLABLE BBOE
BFB0          ; LV # Lo       HANGUL SYLLABLE BBYO
BFCC          ; LV # Lo       HANGUL SYLLABLE BBU
BFE8          ; LV # Lo       HANGUL SYLLABLE BBWEO
C004          ; LV # Lo       HANGUL SYLLABLE BBWE
C020          ; LV # Lo       HANGUL SYLLABLE BBWI
C03C          ; LV # Lo       HANGUL SYLLABLE BBYU
C058          ; LV # Lo       HANGUL SYLLABLE BBEU
C074          ; LV # Lo       HANGUL SYLLABLE BBYI
C090          ; LV # Lo       HANGUL SYLLABLE BBI
C0AC          ; LV # Lo       HANGUL SYLLABLE SA
C0C8          ; LV # Lo       HANGUL SYLLABLE SAE
C0E4          ; LV # Lo       HANGUL SYLLABLE SYA
C100          ; LV # Lo       HANGUL SYLLABLE SYAE
C11C          ; LV # Lo       HANGUL SYLLABLE SEO
C138          ; LV # Lo       HANGUL SYLLABLE SE
C154          ; LV # Lo       HANGUL SYLLABLE SYEO
C170          ; LV # Lo       HANGUL SYLLABLE SYE
C18C          ; LV # Lo       HANGUL SYLLABLE SO
C1A8          ; LV # Lo       HANGUL SYLLABLE SWA
C1C4          ; LV # Lo       HANGUL SYLLABLE SWAE
C1E0          ; LV # Lo       HANGUL SYLLABLE SOE
C1FC          ; LV # Lo       HANGUL SYLLABLE SYO
C218          ; LV # Lo       HANGUL SYLLABLE SU
C234          ; LV # Lo       HANGUL SYLLABLE SWEO
C250          ; LV # Lo       HANGUL SYLLABLE SWE
C26C          ; LV # Lo       HANGUL SYLLABLE SWI
C288          ; LV # Lo       HANGUL SYLLABLE SYU
C2A4          ; LV # Lo       HANGUL SYLLABLE SEU
C2C0          ; LV # Lo       HANGUL SYLLABLE SYI
C2DC          ; LV # Lo       HANGUL SYLLABLE SI
C2F8          ; LV # Lo       HANGUL SYLLABLE SSA
C314          ; LV # Lo       HANGUL SYLLABLE SSAE
C330          ; LV # Lo       HANGUL SYLLABLE SSYA
C34C          ; LV # Lo       HANGUL SYLLABLE SSYAE
C368          ; LV # Lo       HANGUL SYLLABLE SSEO
C384          ; LV # Lo       HANGUL SYLLABLE SSE
C3A0          ; LV # Lo       HANGUL SYLLABLE SSYEO
C3BC          ; LV # Lo       HANGUL SYLLABLE SSYE
C3D8          ; LV # Lo       HANGUL SYLLABLE SSO
C3F4          ; LV # Lo       HANGUL SYLLABLE SSWA
C410          ; LV # Lo       HANGUL SYLLABLE SSWAE
C42C          ; LV # Lo       HANGUL SYLLABLE SSOE
C448          ; LV # Lo       HANGUL SYLLABLE SSYO
C464          ; LV # Lo       HANGUL SYLLABLE SSU
C480          ; LV # Lo       HANGUL SYLLABLE SSWEO
C49C          ; LV # Lo       HANGUL SYLLABLE SSWE
C4B8          ; LV # Lo       HANGUL SYLLABLE SSWI
C4D4          ; LV # Lo       HANGUL SYLLABLE SSYU
C4F0          ; LV # Lo       HANGUL SYLLABLE SSEU
C50C          ; LV # Lo       HANGUL SYLLABLE SSYI
C528          ; LV # Lo       HANGUL SYLLABLE SSI
C544          ; LV # Lo       HANGUL SYLLABLE A
C560          ; LV # Lo       HANGUL SYLLABLE AE
C57C          ; LV # Lo       HANGUL SYLLABLE YA
C598          ; LV # Lo       HANGUL SYLLABLE YAE
C5B4          ; LV # Lo       HANGUL SYLLABLE EO
C5D0          ; LV # Lo       HANGUL SYLLABLE E
C5EC          ; LV # Lo       HANGUL SYLLABLE YEO
C608          ; LV # Lo       HANGUL SYLLABLE YE
C624          ; LV # Lo       HANGUL SYLLABLE O
C640          ; LV # Lo       HANGUL SYLLABLE WA
C65C          ; LV # Lo       HANGUL SYLLABLE WAE
C678          ; LV # Lo       HANGUL SYLLABLE OE
C694          ; LV # Lo       HANGUL SYLLABLE YO
C6B0          ; LV # Lo       HANGUL SYLLABLE U
C6CC          ; LV # Lo       HANGUL SYLLABLE WEO
C6E8          ; LV # Lo       HANGUL SYLLABLE WE
C704          ; LV # Lo       HANGUL SYLLABLE WI
C720          ; LV # Lo       HANGUL SYLLABLE YU
C73C          ; LV # Lo       HANGUL SYLLABLE EU
C758          ; LV # Lo       HANGUL SYLLABLE YI
C774          ; LV # Lo       HANGUL SYLLABLE I
C790          ; LV # Lo       HANGUL SYLLABLE JA
C7AC          ; LV # Lo       HANGUL SYLLABLE JAE
C7C8          ; LV # Lo       HANGUL SYLLABLE JYA
C7E4          ; LV # Lo       HANGUL SYLLABLE JYAE
C800          ; LV # Lo       HANGUL SYLLABLE JEO
C81C          ; LV # Lo       HANGUL SYLLABLE JE
C838          ; LV # Lo       HANGUL SYLLABLE JYEO
C854          ; LV # Lo       HANGUL SYLLABLE JYE
C870          ; LV # Lo       HANGUL SYLLABLE JO
C88C          ; LV # Lo       HANGUL SYLLABLE JWA
C8A8          ; LV # Lo       HANGUL SYLLABLE JWAE
C8C4          ; LV # Lo       HANGUL SYLLABLE JOE
C8E0          ; LV # Lo       HANGUL SYLLABLE JYO
C8FC          ; LV # Lo       HANGUL SYLLABLE JU
C918          ; LV # Lo       HANGUL SYLLABLE JWEO
C934          ; LV # Lo       HANGUL SYLLABLE JWE
C950          ; LV # Lo       HANGUL SYLLABLE JWI
C96C          ; LV # Lo       HANGUL SYLLABLE JYU
C988          ; LV # Lo       HANGUL SYLLABLE JEU
C9A4          ; LV # Lo       HANGUL SYLLABLE JYI
C9C0          ; LV # Lo       HANGUL SYLLABLE JI
C9DC          ; LV # Lo       HANGUL SYLLABLE JJA
C9F8          ; LV # Lo       HANGUL SYLLABLE JJAE
CA14          ; LV # Lo       HANGUL SYLLABLE JJYA
CA30          ; LV # Lo       HANGUL SYLLABLE JJYAE
CA4C          ; LV # Lo       HANGUL SYLLABLE JJEO
CA68          ; LV # Lo       HANGUL SYLLABLE JJE
CA84          ; LV # Lo       HANGUL SYLLABLE JJYEO
CAA0          ; LV # Lo       HANGUL SYLLABLE JJYE
CABC          ; LV # Lo       HANGUL SYLLABLE JJO
CAD8          ; LV # Lo       HANGUL SYLLABLE JJWA
CAF4          ; LV # Lo       HANGUL SYLLABLE JJWAE
CB10          ; LV # Lo       HANGUL SYLLABLE JJOE
CB2C          ; LV # Lo       HANGUL SYLLABLE JJYO
CB48          ; LV # Lo       HANGUL SYLLABLE JJU
CB64          ; LV # Lo       HANGUL SYLLABLE JJWEO
CB80          ; LV # Lo       HANGUL SYLLABLE JJWE
CB9C          ; LV # Lo       HANGUL SYLLABLE JJWI
CBB8          ; LV # Lo       HANGUL SYLLABLE JJYU
CBD4          ; LV # Lo       HANGUL SYLLABLE JJEU
CBF0          ; LV # Lo       HANGUL SYLLABLE JJYI
CC0C          ; LV # Lo       HANGUL SYLLABLE JJI
CC28          ; LV # Lo       HANGUL SYLLABLE CA
CC44          ; LV # Lo       HANGUL SYLLABLE CAE
CC60          ; LV # Lo       HANGUL SYLLABLE CYA
CC7C          ; LV # Lo       HANGUL SYLLABLE CYAE
CC98          ; LV # Lo       HANGUL SYLLABLE CEO
CCB4          ; LV # Lo       HANGUL SYLLABLE CE
CCD0          ; LV # Lo       HANGUL SYLLABLE CYEO
CCEC          ; LV # Lo       HANGUL SYLLABLE CYE
CD08          ; LV # Lo       HANGUL SYLLABLE CO
CD24          ; LV # Lo       HANGUL SYLLABLE CWA
CD40          ; LV # Lo       HANGUL SYLLABLE CWAE
CD5C          ; LV # Lo       HANGUL SYLLABLE COE
CD78          ; LV # Lo       HANGUL SYLLABLE CYO
CD94          ; LV # Lo       HANGUL SYLLABLE CU
CDB0          ; LV # Lo       HANGUL SYLLABLE CWEO
CDCC          ; LV # Lo       HANGUL SYLLABLE CWE
CDE8          ; LV # Lo       HANGUL SYLLABLE CWI
CE04          ; LV # Lo       HANGUL SYLLABLE CYU
CE20          ; LV # Lo       HANGUL SYLLABLE CEU
CE3C          ; LV # Lo       HANGUL SYLLABLE CYI
CE58          ; LV # Lo       HANGUL SYLLABLE CI
CE74          ; LV # Lo       HANGUL SYLLABLE KA
CE90          ; LV # Lo       HANGUL SYLLABLE KAE
CEAC          ; LV # Lo       HANGUL SYLLABLE KYA
CEC8          ; LV # Lo       HANGUL SYLLABLE KYAE
CEE4          ; LV # Lo       HANGUL SYLLABLE KEO
CF00          ; LV # Lo       HANGUL SYLLABLE KE
CF1C          ; LV # Lo       HANGUL SYLLABLE KYEO
CF38          ; LV # Lo       HANGUL SYLLABLE KYE
CF54          ; LV # Lo       HANGUL SYLLABLE KO
CF70          ; LV # Lo       HANGUL SYLLABLE KWA
CF8C          ; LV # Lo       HANGUL SYLLABLE KWAE
CFA8          ; LV # Lo       HANGUL SYLLABLE KOE
CFC4          ; LV # Lo       HANGUL SYLLABLE KYO
CFE0          ; LV # Lo       HANGUL SYLLABLE KU
CFFC          ; LV # Lo       HANGUL SYLLABLE KWEO
D018          ; LV # Lo       HANGUL SYLLABLE KWE
D034          ; LV # Lo       HANGUL SYLLABLE KWI
D050          ; LV # Lo       HANGUL SYLLABLE KYU
D06C          ; LV # Lo       HANGUL SYLLABLE KEU
D088          ; LV # Lo       HANGUL SYLLABLE KYI
D0A4          ; LV # Lo       HANGUL SYLLABLE KI
D0C0          ; LV # Lo       HANGUL SYLLABLE TA
D0DC          ; LV # Lo       HANGUL SYLLABLE TAE
D0F8          ; LV # Lo       HANGUL SYLLABLE TYA
D114          ; LV # Lo       HANGUL SYLLABLE TYAE
D130          ; LV # Lo       HANGUL SYLLABLE TEO
D14C          ; LV # Lo       HANGUL SYLLABLE TE
D168          ; LV # Lo       HANGUL SYLLABLE TYEO
D184          ; LV # Lo       HANGUL SYLLABLE TYE
D1A0          ; LV # Lo       HANGUL SYLLABLE TO
D1BC          ; LV # Lo       HANGUL SYLLABLE TWA
D1D8          ; LV # Lo       HANGUL SYLLABLE TWAE
D1F4          ; LV # Lo       HANGUL SYLLABLE TOE
D210          ; LV # Lo       HANGUL SYLLABLE TYO
D22C          ; LV # Lo       HANGUL SYLLABLE TU
D248          ; LV # Lo       HANGUL SYLLABLE TWEO
D264          ; LV # Lo       HANGUL SYLLABLE TWE
D280          ; LV # Lo       HANGUL SYLLABLE TWI
D29C          ; LV # Lo       HANGUL SYLLABLE TYU
D2B8          ; LV # Lo       HANGUL SYLLABLE TEU
D2D4          ; LV # Lo       HANGUL SYLLABLE TYI
D2F0          ; LV # Lo       HANGUL SYLLABLE TI
D30C          ; LV # Lo       HANGUL SYLLABLE PA
D328          ; LV # Lo       HANGUL SYLLABLE PAE
D344          ; LV # Lo       HANGUL SYLLABLE PYA
D360          ; LV # Lo       HANGUL SYLLABLE PYAE
D37C          ; LV # Lo       HANGUL SYLLABLE PEO
D398          ; LV # Lo       HANGUL SYLLABLE PE
D3B4          ; LV # Lo       HANGUL SYLLABLE PYEO
D3D0          ; LV # Lo       HANGUL SYLLABLE PYE
D3EC          ; LV # Lo       HANGUL SYLLABLE PO
D408          ; LV # Lo       HANGUL SYLLABLE PWA
D424          ; LV # Lo       HANGUL SYLLABLE PWAE
D440          ; LV # Lo       HANGUL SYLLABLE POE
D45C          ; LV # Lo       HANGUL SYLLABLE PYO
D478          ; LV # Lo       HANGUL SYLLABLE PU
D494          ; LV # Lo       HANGUL SYLLABLE PWEO
D4B0          ; LV # Lo       HANGUL SYLLABLE PWE
D4CC          ; LV # Lo       HANGUL SYLLABLE PWI
D4E8          ; LV # Lo       HANGUL SYLLABLE PYU
D504          ; LV # Lo       HANGUL SYLLABLE PEU
D520          ; LV # Lo       HANGUL SYLLABLE PYI
D53C          ; LV # Lo       HANGUL SYLLABLE PI
D558          ; LV # Lo       HANGUL SYLLABLE HA
D574          ; LV # Lo       HANGUL SYLLABLE HAE
D590          ; LV # Lo       HANGUL SYLLABLE HYA
D5AC          ; LV # Lo       HANGUL SYLLABLE HYAE
D5C8          ; LV # Lo       HANGUL SYLLABLE HEO
D5E4          ; LV # Lo       HANGUL SYLLABLE HE
D600          ; LV # Lo       HANGUL SYLLABLE HYEO
D61C          ; LV # Lo       HANGUL SYLLABLE HYE
D638          ; LV # Lo       HANGUL SYLLABLE HO
D654          ; LV # Lo       HANGUL SYLLABLE HWA
D670          ; LV # Lo       HANGUL SYLLABLE HWAE
D68C          ; LV # Lo       HANGUL SYLLABLE HOE
D6A8          ; LV # Lo       HANGUL SYLLABLE HYO
D6C4          ; LV # Lo       HANGUL SYLLABLE HU
D6E0          ; LV # Lo       HANGUL SYLLABLE HWEO
D6FC          ; LV # Lo       HANGUL SYLLABLE HWE
D718          ; LV # Lo       HANGUL SYLLABLE HWI
D734          ; LV # Lo       HANGUL SYLLABLE HYU
D750          ; LV # Lo       HANGUL SYLLABLE HEU
D76C          ; LV # Lo       HANGUL SYLLABLE HYI
D788          ; LV # Lo       HANGUL SYLLABLE HI

# Total code points: 399

# ================================================

AC01..AC1B    ; LVT # Lo  [27] HANGUL SYLLABLE GAG..HANGUL SYLLABLE GAH
AC1D..AC37    ; LVT # Lo  [27] HANGUL SYLLABLE GAEG..HANGUL SYLLABLE GAEH
AC39..AC53    ; LVT # Lo  [27] HANGUL SYLLABLE GYAG..HANGUL SYLLABLE GYAH
AC55..AC6F    ; LVT # Lo  [27] HANGUL SYLLABLE GYAEG..HANGUL SYLLABLE GYAEH
AC71..AC8B    ; LVT # Lo  [27] HANGUL SYLLABLE GEOG..HANGUL SYLLABLE GEOH
AC8D..ACA7    ; LVT # Lo  [27] HANGUL SYLLABLE GEG..HANGUL SYLLABLE GEH
ACA9..ACC3    ; LVT # Lo  [27] HANGUL SYLLABLE GYEOG..HANGUL SYLLABLE GYEOH
ACC5..ACDF    ; LVT # Lo  [27] HANGUL SYLLABLE GYEG..HANGUL SYLLABLE GYEH
ACE1..ACFB    ; LVT # Lo  [27] HANGUL SYLLABLE GOG..HANGUL SYLLABLE GOH
ACFD..AD17    ; LVT # Lo  [27] HANGUL SYLLABLE GWAG..HANGUL SYLLABLE GWAH
AD19..AD33    ; LVT # Lo  [27] HANGUL SYLLABLE GWAEG..HANGUL SYLLABLE GWAEH
AD35..AD4F    ; LVT # Lo  [27] HANGUL SYLLABLE GOEG..HANGUL SYLLABLE GOEH
AD51..AD6B    ; LVT # Lo  [27] HANGUL SYLLABLE GYOG..HANGUL SYLLABLE GYOH
AD6D..AD87    ; LVT # Lo  [27] HANGUL SYLLABLE GUG..HANGUL SYLLABLE GUH
AD89..ADA3    ; LVT # Lo  [27] HANGUL SYLLABLE GWEOG..HANGUL SYLLABLE GWEOH
ADA5..ADBF    ; LVT # Lo  [27] HANGUL SYLLABLE GWEG..HANGUL SYLLABLE GWEH
ADC1..ADDB    ; LVT # Lo  [27] HANGUL SYLLABLE GWIG..HANGUL SYLLABLE GWIH
ADDD..ADF7    ; LVT # Lo  [27] HANGUL SYLLABLE GYUG..HANGUL SYLLABLE GYUH
ADF9..AE13    ; LVT # Lo  [27] HANGUL SYLLABLE GEUG..HANGUL SYLLABLE GEUH
AE15..AE2F    ; LVT # Lo  [27] HANGUL SYLLABLE GYIG..HANGUL SYLLABLE GYIH
AE31..AE4B    ; LVT # Lo  [27] HANGUL SYLLABLE GIG..HANGUL SYLLABLE GIH
AE4D..AE67    ; LVT # Lo  [27] HANGUL SYLLABLE GGAG..HANGUL SYLLABLE GGAH
AE69..AE83    ; LVT # Lo  [27] HANGUL SYLLABLE GGAEG..HANGUL SYLLABLE GGAEH
AE85..AE9F    ; LVT # Lo  [27] HANGUL SYLLABLE GGYAG..HANGUL SYLLABLE GGYAH
AEA1..AEBB    ; LVT # Lo  [27] HANGUL SYLLABLE GGYAEG..HANGUL SYLLABLE GGYAEH
AEBD..AED7    ; LVT # Lo  [27] HANGUL SYLLABLE GGEOG..HANGUL SYLLABLE GGEOH
AED9..AEF3    ; LVT # Lo  [27] HANGUL SYLLABLE GGEG..HANGUL SYLLABLE GGEH
AEF5..AF0F    ; LVT # Lo  [27] HANGUL SYLLABLE GGYEOG..HANGUL SYLLABLE GGYEOH
AF11..AF2B    ; LVT # Lo  [27] HANGUL SYLLABLE GGYEG..HANGUL SYLLABLE GGYEH
AF2D..AF47    ; LVT # Lo  [27] HANGUL SYLLABLE GGOG..HANGUL SYLLABLE GGOH
AF49..AF63    ; LVT # Lo  [27] HANGUL SYLLABLE GGWAG..HANGUL SYLLABLE GGWAH
AF65..AF7F    ; LVT # Lo  [27] HANGUL SYLLABLE GGWAEG..HANGUL SYLLABLE GGWAEH
AF81..AF9B    ; LVT # Lo  [27] HANGUL SYLLABLE GGOEG..HANGUL SYLLABLE GGOEH
AF9D..AFB7    ; LVT # Lo  [27] HANGUL SYLLABLE GGYOG..HANGUL SYLLABLE GGYOH
AFB9..AFD3    ; LVT # Lo  [27] HANGUL SYLLABLE GGUG..HANGUL SYLLABLE GGUH
AFD5..AFEF    ; LVT # Lo  [27] HANGUL SYLLABLE GGWEOG..HANGUL SYLLABLE GGWEOH
AFF1..B00B    ; LVT # Lo  [27] HANGUL SYLLABLE GGWEG..HANGUL SYLLABLE GGWEH
B00D..B027    ; LVT # Lo  [27] HANGUL SYLLABLE GGWIG..HANGUL SYLLABLE GGWIH
B029..B043    ; LVT # Lo  [27] HANGUL SYLLABLE GGYUG..HANGUL SYLLABLE GGYUH
B045..B05F    ; LVT # Lo  [27] HANGUL SYLLABLE GGEUG..HANGUL SYLLABLE GGEUH
B061..B07B    ; LVT # Lo  [27] HANGUL SYLLABLE GGYIG..HANGUL SYLLABLE GGYIH
B07D..B097    ; LVT # Lo  [27] HANGUL SYLLABLE GGIG..HANGUL SYLLABLE GGIH
B099..B0B3    ; LVT # Lo  [27] HANGUL SYLLABLE NAG..HANGUL SYLLABLE NAH
B0B5..B0CF    ; LVT # Lo  [27] HANGUL SYLLABLE NAEG..HANGUL SYLLABLE NAEH
B0D1..B0EB    ; LVT # Lo  [27] HANGUL SYLLABLE NYAG..HANGUL SYLLABLE NYAH
B0ED..B107    ; LVT # Lo  [27] HANGUL SYLLABLE NYAEG..HANGUL SYLLABLE NYAEH
B109..B123    ; LVT # Lo  [27] HANGUL SYLLABLE NEOG..HANGUL SYLLABLE NEOH
B125..B13F    ; LVT # Lo  [27] HANGUL SYLLABLE NEG..HANGUL SYLLABLE NEH
B141..B15B    ; LVT # Lo  [27] HANGUL SYLLABLE NYEOG..HANGUL SYLLABLE NYEOH
B15D..B177    ; LVT # Lo  [27] HANGUL SYLLABLE NYEG..HANGUL SYLLABLE NYEH
B179..B193    ; LVT # Lo  [27] HANGUL SYLLABLE NOG..HANGUL SYLLABLE NOH
B195..B1AF    ; LVT # Lo  [27] HANGUL SYLLABLE NWAG..HANGUL SYLLABLE NWAH
B1B1..B1CB    ; LVT # Lo  [27] HANGUL SYLLABLE NWAEG..HANGUL SYLLABLE NWAEH
B1CD..B1E7    ; LVT # Lo  [27] HANGUL SYLLABLE NOEG..HANGUL SYLLABLE NOEH
B1E9..B203    ; LVT # Lo  [27] HANGUL SYLLABLE NYOG..HANGUL SYLLABLE NYOH
B205..B21F    ; LVT # Lo  [27] HANGUL SYLLABLE NUG..HANGUL SYLLABLE NUH
B221..B23B    ; LVT # Lo  [27] HANGUL SYLLABLE NWEOG..HANGUL SYLLABLE NWEOH
B23D..B257    ; LVT # Lo  [27] HANGUL SYLLABLE NWEG..HANGUL SYLLABLE NWEH
B259..B273    ; LVT # Lo  [27] HANGUL SYLLABLE NWIG..HANGUL SYLLABLE NWIH
B275..B28F    ; LVT # Lo  [27] HANGUL SYLLABLE NYUG..HANGUL SYLLABLE NYUH
B291..B2AB    ; LVT # Lo  [27] HANGUL SYLLABLE NEUG..HANGUL SYLLABLE NEUH
B2AD..B2C7    ; LVT # Lo  [27] HANGUL SYLLABLE NYIG..HANGUL SYLLABLE NYIH
B2C9..B2E3    ; LVT # Lo  [27] HANGUL SYLLABLE NIG..HANGUL SYLLABLE NIH
B2E5..B2FF    ; LVT # Lo  [27] HANGUL SYLLABLE DAG..HANGUL SYLLABLE DAH
B301..B31B    ; LVT # Lo  [27] HANGUL SYLLABLE DAEG..HANGUL SYLLABLE DAEH
B31D..B337    ; LVT # Lo  [27] HANGUL SYLLABLE DYAG..HANGUL SYLLABLE DYAH
B339..B353    ; LVT # Lo  [27] HANGUL SYLLABLE DYAEG..HANGUL SYLLABLE DYAEH
B355..B36F    ; LVT # Lo  [27] HANGUL SYLLABLE DEOG..HANGUL SYLLABLE DEOH
B371..B38B    ; LVT # Lo  [27] HANGUL SYLLABLE DEG..HANGUL SYLLABLE DEH
B38D..B3A7    ; LVT # Lo  [27] HANGUL SYLLABLE DYEOG..HANGUL SYLLABLE DYEOH
B3A9..B3C3    ; LVT # Lo  [27] HANGUL SYLLABLE DYEG..HANGUL SYLLABLE DYEH
B3C5..B3DF    ; LVT # Lo  [27] HANGUL SYLLABLE DOG..HANGUL SYLLABLE DOH
B3E1..B3FB    ; LVT # Lo  [27] HANGUL SYLLABLE DWAG..HANGUL SYLLABLE DWAH
B3FD..B417    ; LVT # Lo  [27] HANGUL SYLLABLE DWAEG..HANGUL SYLLABLE DWAEH
B419..B433    ; LVT # Lo  [27] HANGUL SYLLABLE DOEG..HANGUL SYLLABLE DOEH
B435..B44F    ; LVT # Lo  [27] HANGUL SYLLABLE DYOG..HANGUL SYLLABLE DYOH
B451..B46B    ; LVT # Lo  [27] HANGUL SYLLABLE DUG..HANGUL SYLLABLE DUH
B46D..B487    ; LVT # Lo  [27] HANGUL SYLLABLE DWEOG..HANGUL SYLLABLE DWEOH
B489..B4A3    ; LVT # Lo  [27] HANGUL SYLLABLE DWEG..HANGUL SYLLABLE DWEH
B4A5..B4BF    ; LVT # Lo  [27] HANGUL SYLLABLE DWIG..HANGUL SYLLABLE DWIH
B4C1..B4DB    ; LVT # Lo  [27] HANGUL SYLLABLE DYUG..HANGUL SYLLABLE DYUH
B4DD..B4F7    ; LVT # Lo  [27] HANGUL SYLLABLE DEUG..HANGUL SYLLABLE DEUH
B4F9..B513    ; LVT # Lo  [27] HANGUL SYLLABLE DYIG..HANGUL SYLLABLE DYIH
B515..B52F    ; LVT # Lo  [27] HANGUL SYLLABLE DIG..HANGUL SYLLABLE DIH
B531..B54B    ; LVT # Lo  [27] HANGUL SYLLABLE DDAG..HANGUL SYLLABLE DDAH
B54D..B567    ; LVT # Lo  [27] HANGUL SYLLABLE DDAEG..HANGUL SYLLABLE DDAEH
B569..B583    ; LVT # Lo  [27] HANGUL SYLLABLE DDYAG..HANGUL SYLLABLE DDYAH
B585..B59F    ; LVT # Lo  [27] HANGUL SYLLABLE DDYAEG..HANGUL SYLLABLE DDYAEH
B5A1..B5BB    ; LVT # Lo  [27] HANGUL SYLLABLE DDEOG..HANGUL SYLLABLE DDEOH
B5BD..B5D7    ; LVT # Lo  [27] HANGUL SYLLABLE DDEG..HANGUL SYLLABLE DDEH
B5D9..B5F3    ; LVT # Lo  [27] HANGUL SYLLABLE DDYEOG..HANGUL SYLLABLE DDYEOH
B5F5..B60F    ; LVT # Lo  [27] HANGUL SYLLABLE DDYEG..HANGUL SYLLABLE DDYEH
B611..B62B    ; LVT # Lo  [27] HANGUL SYLLABLE DDOG..HANGUL SYLLABLE DDOH
B62D..B647    ; LVT # Lo  [27] HANGUL SYLLABLE DDWAG..HANGUL SYLLABLE DDWAH
B649..B663    ; LVT # Lo  [27] HANGUL SYLLABLE DDWAEG..HANGUL SYLLABLE DDWAEH
B665..B67F    ; LVT # Lo  [27] HANGUL SYLLABLE DDOEG..HANGUL SYLLABLE DDOEH
B681..B69B    ; LVT # Lo  [27] HANGUL SYLLABLE DDYOG..HANGUL SYLLABLE DDYOH
B69D..B6B7    ; LVT # Lo  [27] HANGUL SYLLABLE DDUG..HANGUL SYLLABLE DDUH
B6B9..B6D3    ; LVT # Lo  [27] HANGUL SYLLABLE DDWEOG..HANGUL SYLLABLE DDWEOH
B6D5..B6EF    ; LVT # Lo  [27] HANGUL SYLLABLE DDWEG..HANGUL SYLLABLE DDWEH
B6F1..B70B    ; LVT # Lo  [27] HANGUL SYLLABLE DDWIG..HANGUL SYLLABLE DDWIH
B70D..B727    ; LVT # Lo  [27] HANGUL SYLLABLE DDYUG..HANGUL SYLLABLE DDYUH
B729..B743    ; LVT # Lo  [27] HANGUL SYLLABLE DDEUG..HANGUL SYLLABLE DDEUH
B745..B75F    ; LVT # Lo  [27] HANGUL SYLLABLE DDYIG..HANGUL SYLLABLE DDYIH
B761..B77B    ; LVT # Lo  [27] HANGUL SYLLABLE DDIG..HANGUL SYLLABLE DDIH
B77D..B797    ; LVT # Lo  [27] HANGUL SYLLABLE RAG..HANGUL SYLLABLE RAH
B799..B7B3    ; LVT # Lo  [27] HANGUL SYLLABLE RAEG..HANGUL SYLLABLE RAEH
B7B5..B7CF    ; LVT # Lo  [27] HANGUL SYLLABLE RYAG..HANGUL SYLLABLE RYAH
B7D1..B7EB    ; LVT # Lo  [27] HANGUL SYLLABLE RYAEG..HANGUL SYLLABLE RYAEH
B7ED..B807    ; LVT # Lo  [27] HANGUL SYLLABLE REOG..HANGUL SYLLABLE REOH
B809..B823    ; LVT # Lo  [27] HANGUL SYLLABLE REG..HANGUL SYLLABLE REH
B825..B83F    ; LVT # Lo  [27] HANGUL SYLLABLE RYEOG..HANGUL SYLLABLE RYEOH
B841..B85B    ; LVT # Lo  [27] HANGUL SYLLABLE RYEG..HANGUL SYLLABLE RYEH
B85D..B877    ; LVT # Lo  [27] HANGUL SYLLABLE ROG..HANGUL SYLLABLE ROH
B879..B893    ; LVT # Lo  [27] HANGUL SYLLABLE RWAG..HANGUL SYLLABLE RWAH
B895..B8AF    ; LVT # Lo  [27] HANGUL SYLLABLE RWAEG..HANGUL SYLLABLE RWAEH
B8B1..B8CB    ; LVT # Lo  [27] HANGUL SYLLABLE ROEG..HANGUL SYLLABLE ROEH
B8CD..B8E7    ; LVT # Lo  [27] HANGUL SYLLABLE RYOG..HANGUL SYLLABLE RYOH
B8E9..B903    ; LVT # Lo  [27] HANGUL SYLLABLE RUG..HANGUL SYLLABLE RUH
B905..B91F    ; LVT # Lo  [27] HANGUL SYLLABLE RWEOG..HANGUL SYLLABLE RWEOH
B921..B93B    ; LVT # Lo  [27] HANGUL SYLLABLE RWEG..HANGUL SYLLABLE RWEH
B93D..B957    ; LVT # Lo  [27] HANGUL SYLLABLE RWIG..HANGUL SYLLABLE RWIH
B959..B973    ; LVT # Lo  [27] HANGUL SYLLABLE RYUG..HANGUL SYLLABLE RYUH
B975..B98F    ; LVT # Lo  [27] HANGUL SYLLABLE REUG..HANGUL SYLLABLE REUH
B991..B9AB    ; LVT # Lo  [27] HANGUL SYLLABLE RYIG..HANGUL SYLLABLE RYIH
B9AD..B9C7    ; LVT # Lo  [27] HANGUL SYLLABLE RIG..HANGUL SYLLABLE RIH
B9C9..B9E3    ; LVT # Lo  [27] HANGUL SYLLABLE MAG..HANGUL SYLLABLE MAH
B9E5..B9FF    ; LVT # Lo  [27] HANGUL SYLLABLE MAEG..HANGUL SYLLABLE MAEH
BA01..BA1B    ; LVT # Lo  [27] HANGUL SYLLABLE MYAG..HANGUL SYLLABLE MYAH
BA1D..BA37    ; LVT # Lo  [27] HANGUL SYLLABLE MYAEG..HANGUL SYLLABLE MYAEH
BA39..BA53    ; LVT # Lo  [27] HANGUL SYLLABLE MEOG..HANGUL SYLLABLE MEOH
BA55..BA6F    ; LVT # Lo  [27] HANGUL SYLLABLE MEG..HANGUL SYLLABLE MEH
BA71..BA8B    ; LVT # Lo  [27] HANGUL SYLLABLE MYEOG..HANGUL SYLLABLE MYEOH
BA8D..BAA7    ; LVT # Lo  [27] HANGUL SYLLABLE MYEG..HANGUL SYLLABLE MYEH
BAA9..BAC3    ; LVT # Lo  [27] HANGUL SYLLABLE MOG..HANGUL SYLLABLE MOH
BAC5..BADF    ; LVT # Lo  [27] HANGUL SYLLABLE MWAG..HANGUL SYLLABLE MWAH
BAE1..BAFB    ; LVT # Lo  [27] HANGUL SYLLABLE MWAEG..HANGUL SYLLABLE MWAEH
BAFD..BB17    ; LVT # Lo  [27] HANGUL SYLLABLE MOEG..HANGUL SYLLABLE MOEH
BB19..BB33    ; LVT # Lo  [27] HANGUL SYLLABLE MYOG..HANGUL SYLLABLE MYOH
BB35..BB4F    ; LVT # Lo  [27] HANGUL SYLLABLE MUG..HANGUL SYLLABLE MUH
BB51..BB6B    ; LVT # Lo  [27] HANGUL SYLLABLE MWEOG..HANGUL SYLLABLE MWEOH
BB6D..BB87    ; LVT # Lo  [27] HANGUL SYLLABLE MWEG..HANGUL SYLLABLE MWEH
BB89..BBA3    ; LVT # Lo  [27] HANGUL SYLLABLE MWIG..HANGUL SYLLABLE MWIH
BBA5..BBBF    ; LVT # Lo  [27] HANGUL SYLLABLE MYUG..HANGUL SYLLABLE MYUH
BBC1..BBDB    ; LVT # Lo  [27] HANGUL SYLLABLE MEUG..HANGUL SYLLABLE MEUH
BBDD..BBF7    ; LVT # Lo  [27] HANGUL SYLLABLE MYIG..HANGUL SYLLABLE MYIH
BBF9..BC13    ; LVT # Lo  [27] HANGUL SYLLABLE MIG..HANGUL SYLLABLE MIH
BC15..BC2F    ; LVT # Lo  [27] HANGUL SYLLABLE BAG..HANGUL SYLLABLE BAH
BC31..BC4B    ; LVT # Lo  [27] HANGUL SYLLABLE BAEG..HANGUL SYLLABLE BAEH
BC4D..BC67    ; LVT # Lo  [27] HANGUL SYLLABLE BYAG..HANGUL SYLLABLE BYAH
BC69..BC83    ; LVT # Lo  [27] HANGUL SYLLABLE BYAEG..HANGUL SYLLABLE BYAEH
BC85..BC9F    ; LVT # Lo  [27] HANGUL SYLLABLE BEOG..HANGUL SYLLABLE BEOH
BCA1..BCBB    ; LVT # Lo  [27] HANGUL SYLLABLE BEG..HANGUL SYLLABLE BEH
BCBD..BCD7    ; LVT # Lo  [27] HANGUL SYLLABLE BYEOG..HANGUL SYLLABLE BYEOH
BCD9..BCF3    ; LVT # Lo  [27] HANGUL SYLLABLE BYEG..HANGUL SYLLABLE BYEH
BCF5..BD0F    ; LVT # Lo  [27] HANGUL SYLLABLE BOG..HANGUL SYLLABLE BOH
BD11..BD2B    ; LVT # Lo  [27] HANGUL SYLLABLE BWAG..HANGUL SYLLABLE BWAH
BD2D..BD47    ; LVT # Lo  [27] HANGUL SYLLABLE BWAEG..HANGUL SYLLABLE BWAEH
BD49..BD63    ; LVT # Lo  [27] HANGUL SYLLABLE BOEG..HANGUL SYLLABLE BOEH
BD65..BD7F    ; LVT # Lo  [27] HANGUL SYLLABLE BYOG..HANGUL SYLLABLE BYOH
BD81..BD9B    ; LVT # Lo  [27] HANGUL SYLLABLE BUG..HANGUL SYLLABLE BUH
BD9D..BDB7    ; LVT # Lo  [27] HANGUL SYLLABLE BWEOG..HANGUL SYLLABLE BWEOH
BDB9..BDD3    ; LVT # Lo  [27] HANGUL SYLLABLE BWEG..HANGUL SYLLABLE BWEH
BDD5..BDEF    ; LVT # Lo  [27] HANGUL SYLLABLE BWIG..HANGUL SYLLABLE BWIH
BDF1..BE0B    ; LVT # Lo  [27] HANGUL SYLLABLE BYUG..HANGUL SYLLABLE BYUH
BE0D..BE27    ; LVT # Lo  [27] HANGUL SYLLABLE BEUG..HANGUL SYLLABLE BEUH
BE29..BE43    ; LVT # Lo  [27] HANGUL SYLLABLE BYIG..HANGUL SYLLABLE BYIH
BE45..BE5F    ; LVT # Lo  [27] HANGUL SYLLABLE BIG..HANGUL SYLLABLE BIH
BE61..BE7B    ; LVT # Lo  [27] HANGUL SYLLABLE BBAG..HANGUL SYLLABLE BBAH
BE7D..BE97    ; LVT # Lo  [27] HANGUL SYLLABLE BBAEG..HANGUL SYLLABLE BBAEH
BE99..BEB3    ; LVT # Lo  [27] HANGUL SYLLABLE BBYAG..HANGUL SYLLABLE BBYAH
BEB5..BECF    ; LVT # Lo  [27] HANGUL SYLLABLE BBYAEG..HANGUL SYLLABLE BBYAEH
BED1..BEEB    ; LVT # Lo  [27] HANGUL SYLLABLE BBEOG..HANGUL SYLLABLE BBEOH
BEED..BF07    ; LVT # Lo  [27] HANGUL SYLLABLE BBEG..HANGUL SYLLABLE BBEH
BF09..BF23    ; LVT # Lo  [27] HANGUL SYLLABLE BBYEOG..HANGUL SYLLABLE BBYEOH
BF25..BF3F    ; LVT # Lo  [27] HANGUL SYLLABLE BBYEG..HANGUL SYLLABLE BBYEH
BF41..BF5B    ; LVT # Lo  [27] HANGUL SYLLABLE BBOG..HANGUL SYLLABLE BBOH
BF5D..BF77    ; LVT # Lo  [27] HANGUL SYLLABLE BBWAG..HANGUL SYLLABLE BBWAH
BF79..BF93    ; LVT # Lo  [27] HANGUL SYLLABLE BBWAEG..HANGUL SYLLABLE BBWAEH
BF95..BFAF    ; LVT # Lo  [27] HANGUL SYLLABLE BBOEG..HANGUL SYLLABLE BBOEH
BFB1..BFCB    ; LVT # Lo  [27] HANGUL SYLLABLE BBYOG..HANGUL SYLLABLE BBYOH
BFCD..BFE7    ; LVT # Lo  [27] HANGUL SYLLABLE BBUG..HANGUL SYLLABLE BBUH
BFE9..C003    ; LVT # Lo  [27] HANGUL SYLLABLE BBWEOG..HANGUL SYLLABLE BBWEOH
C005..C01F    ; LVT # Lo  [27] HANGUL SYLLABLE BBWEG..HANGUL SYLLABLE BBWEH
C021..C03B    ; LVT # Lo  [27] HANGUL SYLLABLE BBWIG..HANGUL SYLLABLE BBWIH
C03D..C057    ; LVT # Lo  [27] HANGUL SYLLABLE BBYUG..HANGUL SYLLABLE BBYUH
C059..C073    ; LVT # Lo  [27] HANGUL SYLLABLE BBEUG..HANGUL SYLLABLE BBEUH
C075..C08F    ; LVT # Lo  [27] HANGUL SYLLABLE BBYIG..HANGUL SYLLABLE BBYIH
C091..C0AB    ; LVT # Lo  [27] HANGUL SYLLABLE BBIG..HANGUL SYLLABLE BBIH
C0AD..C0C7    ; LVT # Lo  [27] HANGUL SYLLABLE SAG..HANGUL SYLLABLE SAH
C0C9..C0E3    ; LVT # Lo  [27] HANGUL SYLLABLE SAEG..HANGUL SYLLABLE SAEH
C0E5..C0FF    ; LVT # Lo  [27] HANGUL SYLLABLE SYAG..HANGUL SYLLABLE SYAH
C101..C11B    ; LVT # Lo  [27] HANGUL SYLLABLE SYAEG..HANGUL SYLLABLE SYAEH
C11D..C137    ; LVT # Lo  [27] HANGUL SYLLABLE SEOG..HANGUL SYLLABLE SEOH
C139..C153    ; LVT # Lo  [27] HANGUL SYLLABLE SEG..HANGUL SYLLABLE SEH
C155..C16F    ; LVT # Lo  [27] HANGUL SYLLABLE SYEOG..HANGUL SYLLABLE SYEOH
C171..C18B    ; LVT # Lo  [27] HANGUL SYLLABLE SYEG..HANGUL SYLLABLE SYEH
C18D..C1A7    ; LVT # Lo  [27] HANGUL SYLLABLE SOG..HANGUL SYLLABLE SOH
C1A9..C1C3    ; LVT # Lo  [27] HANGUL SYLLABLE SWAG..HANGUL SYLLABLE SWAH
C1C5..C1DF    ; LVT # Lo  [27] HANGUL SYLLABLE SWAEG..HANGUL SYLLABLE SWAEH
C1E1..C1FB    ; LVT # Lo  [27] HANGUL SYLLABLE SOEG..HANGUL SYLLABLE SOEH
C1FD..C217    ; LVT # Lo  [27] HANGUL SYLLABLE SYOG..HANGUL SYLLABLE SYOH
C219..C233    ; LVT # Lo  [27] HANGUL SYLLABLE SUG..HANGUL SYLLABLE SUH
C235..C24F    ; LVT # Lo  [27] HANGUL SYLLABLE SWEOG..HANGUL SYLLABLE SWEOH
C251..C26B    ; LVT # Lo  [27] HANGUL SYLLABLE SWEG..HANGUL SYLLABLE SWEH
C26D..C287    ; LVT # Lo  [27] HANGUL SYLLABLE SWIG..HANGUL SYLLABLE SWIH
C289..C2A3    ; LVT # Lo  [27] HANGUL SYLLABLE SYUG..HANGUL SYLLABLE SYUH
C2A5..C2BF    ; LVT # Lo  [27] HANGUL SYLLABLE SEUG..HANGUL SYLLABLE SEUH
C2C1..C2DB    ; LVT # Lo  [27] HANGUL SYLLABLE SYIG..HANGUL SYLLABLE SYIH
C2DD..C2F7    ; LVT # Lo  [27] HANGUL SYLLABLE SIG..HANGUL SYLLABLE SIH
C2F9..C313    ; LVT # Lo  [27] HANGUL SYLLABLE SSAG..HANGUL SYLLABLE SSAH
C315..C32F    ; LVT # Lo  [27] HANGUL SYLLABLE SSAEG..HANGUL SYLLABLE SSAEH
C331..C34B    ; LVT # Lo  [27] HANGUL SYLLABLE SSYAG..HANGUL SYLLABLE SSYAH
C34D..C367    ; LVT # Lo  [27] HANGUL SYLLABLE SSYAEG..HANGUL SYLLABLE SSYAEH
C369..C383    ; LVT # Lo  [27] HANGUL SYLLABLE SSEOG..HANGUL SYLLABLE SSEOH
C385..C39F    ; LVT # Lo  [27] HANGUL SYLLABLE SSEG..HANGUL SYLLABLE SSEH
C3A1..C3BB    ; LVT # Lo  [27] HANGUL SYLLABLE SSYEOG..HANGUL SYLLABLE SSYEOH
C3BD..C3D7    ; LVT # Lo  [27] HANGUL SYLLABLE SSYEG..HANGUL SYLLABLE SSYEH
C3D9..C3F3    ; LVT # Lo  [27] HANGUL SYLLABLE SSOG..HANGUL SYLLABLE SSOH
C3F5..C40F    ; LVT # Lo  [27] HANGUL SYLLABLE SSWAG..HANGUL SYLLABLE SSWAH
C411..C42B    ; LVT # Lo  [27] HANGUL SYLLABLE SSWAEG..HANGUL SYLLABLE SSWAEH
C42D..C447    ; LVT # Lo  [27] HANGUL SYLLABLE SSOEG..HANGUL SYLLABLE SSOEH
C449..C463    ; LVT # Lo  [27] HANGUL SYLLABLE SSYOG..HANGUL SYLLABLE SSYOH
C465..C47F    ; LVT # Lo  [27] HANGUL SYLLABLE SSUG..HANGUL SYLLABLE SSUH
C481..C49B    ; LVT # Lo  [27] HANGUL SYLLABLE SSWEOG..HANGUL SYLLABLE SSWEOH
C49D..C4B7    ; LVT # Lo  [27] HANGUL SYLLABLE SSWEG..HANGUL SYLLABLE SSWEH
C4B9..C4D3    ; LVT # Lo  [27] HANGUL SYLLABLE SSWIG..HANGUL SYLLABLE SSWIH
C4D5..C4EF    ; LVT # Lo  [27] HANGUL SYLLABLE SSYUG..HANGUL SYLLABLE SSYUH
C4F1..C50B    ; LVT # Lo  [27] HANGUL SYLLABLE SSEUG..HANGUL SYLLABLE SSEUH
C50D..C527    ; LVT # Lo  [27] HANGUL SYLLABLE SSYIG..HANGUL SYLLABLE SSYIH
C529..C543    ; LVT # Lo  [27] HANGUL SYLLABLE SSIG..HANGUL SYLLABLE SSIH
C545..C55F    ; LVT # Lo  [27] HANGUL SYLLABLE AG..HANGUL SYLLABLE AH
C561..C57B    ; LVT # Lo  [27] HANGUL SYLLABLE AEG..HANGUL SYLLABLE AEH
C57D..C597    ; LVT # Lo  [27] HANGUL SYLLABLE YAG..HANGUL SYLLABLE YAH
C599..C5B3    ; LVT # Lo  [27] HANGUL SYLLABLE YAEG..HANGUL SYLLABLE YAEH
C5B5..C5CF    ; LVT # Lo  [27] HANGUL SYLLABLE EOG..HANGUL SYLLABLE EOH
C5D1..C5EB    ; LVT # Lo  [27] HANGUL SYLLABLE EG..HANGUL SYLLABLE EH
C5ED..C607    ; LVT # Lo  [27] HANGUL SYLLABLE YEOG..HANGUL SYLLABLE YEOH
C609..C623    ; LVT # Lo  [27] HANGUL SYLLABLE YEG..HANGUL SYLLABLE YEH
C625..C63F    ; LVT # Lo  [27] HANGUL SYLLABLE OG..HANGUL SYLLABLE OH
C641..C65B    ; LVT # Lo  [27] HANGUL SYLLABLE WAG..HANGUL SYLLABLE WAH
C65D..C677    ; LVT # Lo  [27] HANGUL SYLLABLE WAEG..HANGUL SYLLABLE WAEH
C679..C693    ; LVT # Lo  [27] HANGUL SYLLABLE OEG..HANGUL SYLLABLE OEH
C695..C6AF    ; LVT # Lo  [27] HANGUL SYLLABLE YOG..HANGUL SYLLABLE YOH
C6B1..C6CB    ; LVT # Lo  [27] HANGUL SYLLABLE UG..HANGUL SYLLABLE UH
C6CD..C6E7    ; LVT # Lo  [27] HANGUL SYLLABLE WEOG..HANGUL SYLLABLE WEOH
C6E9..C703    ; LVT # Lo  [27] HANGUL SYLLABLE WEG..HANGUL SYLLABLE WEH
C705..C71F    ; LVT # Lo  [27] HANGUL SYLLABLE WIG..HANGUL SYLLABLE WIH
C721..C73B    ; LVT # Lo  [27] HANGUL SYLLABLE YUG..HANGUL SYLLABLE YUH
C73D..C757    ; LVT # Lo  [27] HANGUL SYLLABLE EUG..HANGUL SYLLABLE EUH
C759..C773    ; LVT # Lo  [27] HANGUL SYLLABLE YIG..HANGUL SYLLABLE YIH
C775..C78F    ; LVT # Lo  [27] HANGUL SYLLABLE IG..HANGUL SYLLABLE IH
C791..C7AB    ; LVT # Lo  [27] HANGUL SYLLABLE JAG..HANGUL SYLLABLE JAH
C7AD..C7C7    ; LVT # Lo  [27] HANGUL SYLLABLE JAEG..HANGUL SYLLABLE JAEH
C7C9..C7E3    ; LVT # Lo  [27] HANGUL SYLLABLE JYAG..HANGUL SYLLABLE JYAH
C7E5..C7FF    ; LVT # Lo  [27] HANGUL SYLLABLE JYAEG..HANGUL SYLLABLE JYAEH
C801..C81B    ; LVT # Lo  [27] HANGUL SYLLABLE JEOG..HANGUL SYLLABLE JEOH
C81D..C837    ; LVT # Lo  [27] HANGUL SYLLABLE JEG..HANGUL SYLLABLE JEH
C839..C853    ; LVT # Lo  [27] HANGUL SYLLABLE JYEOG..HANGUL SYLLABLE JYEOH
C855..C86F    ; LVT # Lo  [27] HANGUL SYLLABLE JYEG..HANGUL SYLLABLE JYEH
C871..C88B    ; LVT # Lo  [27] HANGUL SYLLABLE JOG..HANGUL SYLLABLE JOH
C88D..C8A7    ; LVT # Lo  [27] HANGUL SYLLABLE JWAG..HANGUL SYLLABLE JWAH
C8A9..C8C3    ; LVT # Lo  [27] HANGUL SYLLABLE JWAEG..HANGUL SYLLABLE JWAEH
C8C5..C8DF    ; LVT # Lo  [27] HANGUL SYLLABLE JOEG..HANGUL SYLLABLE JOEH
C8E1..C8FB    ; LVT # Lo  [27] HANGUL SYLLABLE JYOG..HANGUL SYLLABLE JYOH
C8FD..C917    ; LVT # Lo  [27] HANGUL SYLLABLE JUG..HANGUL SYLLABLE JUH
C919..C933    ; LVT # Lo  [27] HANGUL SYLLABLE JWEOG..HANGUL SYLLABLE JWEOH
C935..C94F    ; LVT # Lo  [27] HANGUL SYLLABLE JWEG..HANGUL SYLLABLE JWEH
C951..C96B    ; LVT # Lo  [27] HANGUL SYLLABLE JWIG..HANGUL SYLLABLE JWIH
C96D..C987    ; LVT # Lo  [27] HANGUL SYLLABLE JYUG..HANGUL SYLLABLE JYUH
C989..C9A3    ; LVT # Lo  [27] HANGUL SYLLABLE JEUG..HANGUL SYLLABLE JEUH
C9A5..C9BF    ; LVT # Lo  [27] HANGUL SYLLABLE JYIG..HANGUL SYLLABLE JYIH
C9C1..C9DB    ; LVT # Lo  [27] HANGUL SYLLABLE JIG..HANGUL SYLLABLE JIH
C9DD..C9F7    ; LVT # Lo  [27] HANGUL SYLLABLE JJAG..HANGUL SYLLABLE JJAH
C9F9..CA13    ; LVT # Lo  [27] HANGUL SYLLABLE JJAEG..HANGUL SYLLABLE JJAEH
CA15..CA2F    ; LVT # Lo  [27] HANGUL SYLLABLE JJYAG..HANGUL SYLLABLE JJYAH
CA31..CA4B    ; LVT # Lo  [27] HANGUL SYLLABLE JJYAEG..HANGUL SYLLABLE JJYAEH
CA4D..CA67    ; LVT # Lo  [27] HANGUL SYLLABLE JJEOG..HANGUL SYLLABLE JJEOH
CA69..CA83    ; LVT # Lo  [27] HANGUL SYLLABLE JJEG..HANGUL SYLLABLE JJEH
CA85..CA9F    ; LVT # Lo  [27] HANGUL SYLLABLE JJYEOG..HANGUL SYLLABLE JJYEOH
CAA1..CABB    ; LVT # Lo  [27] HANGUL SYLLABLE JJYEG..HANGUL SYLLABLE JJYEH
CABD..CAD7    ; LVT # Lo  [27] HANGUL SYLLABLE JJOG..HANGUL SYLLABLE JJOH
CAD9..CAF3    ; LVT # Lo  [27] HANGUL SYLLABLE JJWAG..HANGUL SYLLABLE JJWAH
CAF5..CB0F    ; LVT # Lo  [27] HANGUL SYLLABLE JJWAEG..HANGUL SYLLABLE JJWAEH
CB11..CB2B    ; LVT # Lo  [27] HANGUL SYLLABLE JJOEG..HANGUL SYLLABLE JJOEH
CB2D..CB47    ; LVT # Lo  [27] HANGUL SYLLABLE JJYOG..HANGUL SYLLABLE JJYOH
CB49..CB63    ; LVT # Lo  [27] HANGUL SYLLABLE JJUG..HANGUL SYLLABLE JJUH
CB65..CB7F    ; LVT # Lo  [27] HANGUL SYLLABLE JJWEOG..HANGUL SYLLABLE JJWEOH
CB81..CB9B    ; LVT # Lo  [27] HANGUL SYLLABLE JJWEG..HANGUL SYLLABLE JJWEH
CB9D..CBB7    ; LVT # Lo  [27] HANGUL SYLLABLE JJWIG..HANGUL SYLLABLE JJWIH
CBB9..CBD3    ; LVT # Lo  [27] HANGUL SYLLABLE JJYUG..HANGUL SYLLABLE JJYUH
CBD5..CBEF    ; LVT # Lo  [27] HANGUL SYLLABLE JJEUG..HANGUL SYLLABLE JJEUH
CBF1..CC0B    ; LVT # Lo  [27] HANGUL SYLLABLE JJYIG..HANGUL SYLLABLE JJYIH
CC0D..CC27    ; LVT # Lo  [27] HANGUL SYLLABLE JJIG..HANGUL SYLLABLE JJIH
CC29..CC43    ; LVT # Lo  [27] HANGUL SYLLABLE CAG..HANGUL SYLLABLE CAH
CC45..CC5F    ; LVT # Lo  [27] HANGUL SYLLABLE CAEG..HANGUL SYLLABLE CAEH
CC61..CC7B    ; LVT # Lo  [27] HANGUL SYLLABLE CYAG..HANGUL SYLLABLE CYAH
CC7D..CC97    ; LVT # Lo  [27] HANGUL SYLLABLE CYAEG..HANGUL SYLLABLE CYAEH
CC99..CCB3    ; LVT # Lo  [27] HANGUL SYLLABLE CEOG..HANGUL SYLLABLE CEOH
CCB5..CCCF    ; LVT # Lo  [27] HANGUL SYLLABLE CEG..HANGUL SYLLABLE CEH
CCD1..CCEB    ; LVT # Lo  [27] HANGUL SYLLABLE CYEOG..HANGUL SYLLABLE CYEOH
CCED..CD07    ; LVT # Lo  [27] HANGUL SYLLABLE CYEG..HANGUL SYLLABLE CYEH
CD09..CD23    ; LVT # Lo  [27] HANGUL SYLLABLE COG..HANGUL SYLLABLE COH
CD25..CD3F    ; LVT # Lo  [27] HANGUL SYLLABLE CWAG..HANGUL SYLLABLE CWAH
CD41..CD5B    ; LVT # Lo  [27] HANGUL SYLLABLE CWAEG..HANGUL SYLLABLE CWAEH
CD5D..CD77    ; LVT # Lo  [27] HANGUL SYLLABLE COEG..HANGUL SYLLABLE COEH
CD79..CD93    ; LVT # Lo  [27] HANGUL SYLLABLE CYOG..HANGUL SYLLABLE CYOH
CD95..CDAF    ; LVT # Lo  [27] HANGUL SYLLABLE CUG..HANGUL SYLLABLE CUH
CDB1..CDCB    ; LVT # Lo  [27] HANGUL SYLLABLE CWEOG..HANGUL SYLLABLE CWEOH
CDCD..CDE7    ; LVT # Lo  [27] HANGUL SYLLABLE CWEG..HANGUL SYLLABLE CWEH
CDE9..CE03    ; LVT # Lo  [27] HANGUL SYLLABLE CWIG..HANGUL SYLLABLE CWIH
CE05..CE1F    ; LVT # Lo  [27] HANGUL SYLLABLE CYUG..HANGUL SYLLABLE CYUH
CE21..CE3B    ; LVT # Lo  [27] HANGUL SYLLABLE CEUG..HANGUL SYLLABLE CEUH
CE3D..CE57    ; LVT # Lo  [27] HANGUL SYLLABLE CYIG..HANGUL SYLLABLE CYIH
CE59..CE73    ; LVT # Lo  [27] HANGUL SYLLABLE CIG..HANGUL SYLLABLE CIH
CE75..CE8F    ; LVT # Lo  [27] HANGUL SYLLABLE KAG..HANGUL SYLLABLE KAH
CE91..CEAB    ; LVT # Lo  [27] HANGUL SYLLABLE KAEG..HANGUL SYLLABLE KAEH
CEAD..CEC7    ; LVT # Lo  [27] HANGUL SYLLABLE KYAG..HANGUL SYLLABLE KYAH
CEC9..CEE3    ; LVT # Lo  [27] HANGUL SYLLABLE KYAEG..HANGUL SYLLABLE KYAEH
CEE5..CEFF    ; LVT # Lo  [27] HANGUL SYLLABLE KEOG..HANGUL SYLLABLE KEOH
CF01..CF1B    ; LVT # Lo  [27] HANGUL SYLLABLE KEG..HANGUL SYLLABLE KEH
CF1D..CF37    ; LVT # Lo  [27] HANGUL SYLLABLE KYEOG..HANGUL SYLLABLE KYEOH
CF39..CF53    ; LVT # Lo  [27] HANGUL SYLLABLE KYEG..HANGUL SYLLABLE KYEH
CF55..CF6F    ; LVT # Lo  [27] HANGUL SYLLABLE KOG..HANGUL SYLLABLE KOH
CF71..CF8B    ; LVT # Lo  [27] HANGUL SYLLABLE KWAG..HANGUL SYLLABLE KWAH
CF8D..CFA7    ; LVT # Lo  [27] HANGUL SYLLABLE KWAEG..HANGUL SYLLABLE KWAEH
CFA9..CFC3    ; LVT # Lo  [27] HANGUL SYLLABLE KOEG..HANGUL SYLLABLE KOEH
CFC5..CFDF    ; LVT # Lo  [27] HANGUL SYLLABLE KYOG..HANGUL SYLLABLE KYOH
CFE1..CFFB    ; LVT # Lo  [27] HANGUL SYLLABLE KUG..HANGUL SYLLABLE KUH
CFFD..D017    ; LVT # Lo  [27] HANGUL SYLLABLE KWEOG..HANGUL SYLLABLE KWEOH
D019..D033    ; LVT # Lo  [27] HANGUL SYLLABLE KWEG..HANGUL SYLLABLE KWEH
D035..D04F    ; LVT # Lo  [27] HANGUL SYLLABLE KWIG..HANGUL SYLLABLE KWIH
D051..D06B    ; LVT # Lo  [27] HANGUL SYLLABLE KYUG..HANGUL SYLLABLE KYUH
D06D..D087    ; LVT # Lo  [27] HANGUL SYLLABLE KEUG..HANGUL SYLLABLE KEUH
D089..D0A3    ; LVT # Lo  [27] HANGUL SYLLABLE KYIG..HANGUL SYLLABLE KYIH
D0A5..D0BF    ; LVT # Lo  [27] HANGUL SYLLABLE KIG..HANGUL SYLLABLE KIH
D0C1..D0DB    ; LVT # Lo  [27] HANGUL SYLLABLE TAG..HANGUL SYLLABLE TAH
D0DD..D0F7    ; LVT # Lo  [27] HANGUL SYLLABLE TAEG..HANGUL SYLLABLE TAEH
D0F9..D113    ; LVT # Lo  [27] HANGUL SYLLABLE TYAG..HANGUL SYLLABLE TYAH
D115..D12F    ; LVT # Lo  [27] HANGUL SYLLABLE TYAEG..HANGUL SYLLABLE TYAEH
D131..D14B    ; LVT # Lo  [27] HANGUL SYLLABLE TEOG..HANGUL SYLLABLE TEOH
D14D..D167    ; LVT # Lo  [27] HANGUL SYLLABLE TEG..HANGUL SYLLABLE TEH
D169..D183    ; LVT # Lo  [27] HANGUL SYLLABLE TYEOG..HANGUL SYLLABLE TYEOH
D185..D19F    ; LVT # Lo  [27] HANGUL SYLLABLE TYEG..HANGUL SYLLABLE TYEH
D1A1..D1BB    ; LVT # Lo  [27] HANGUL SYLLABLE TOG..HANGUL SYLLABLE TOH
D1BD..D1D7    ; LVT # Lo  [27] HANGUL SYLLABLE TWAG..HANGUL SYLLABLE TWAH
D1D9..D1F3    ; LVT # Lo  [27] HANGUL SYLLABLE TWAEG..HANGUL SYLLABLE TWAEH
D1F5..D20F    ; LVT # Lo  [27] HANGUL SYLLABLE TOEG..HANGUL SYLLABLE TOEH
D211..D22B    ; LVT # Lo  [27] HANGUL SYLLABLE TYOG..HANGUL SYLLABLE TYOH
D22D..D247    ; LVT # Lo  [27] HANGUL SYLLABLE TUG..HANGUL SYLLABLE TUH
D249..D263    ; LVT # Lo  [27] HANGUL SYLLABLE TWEOG..HANGUL SYLLABLE TWEOH
D265..D27F    ; LVT # Lo  [27] HANGUL SYLLABLE TWEG..HANGUL SYLLABLE TWEH
D281..D29B    ; LVT # Lo  [27] HANGUL SYLLABLE TWIG..HANGUL SYLLABLE TWIH
D29D..D2B7    ; LVT # Lo  [27] HANGUL SYLLABLE TYUG..HANGUL SYLLABLE TYUH
D2B9..D2D3    ; LVT # Lo  [27] HANGUL SYLLABLE TEUG..HANGUL SYLLABLE TEUH
D2D5..D2EF    ; LVT # Lo  [27] HANGUL SYLLABLE TYIG..HANGUL SYLLABLE TYIH
D2F1..D30B    ; LVT # Lo  [27] HANGUL SYLLABLE TIG..HANGUL SYLLABLE TIH
D30D..D327    ; LVT # Lo  [27] HANGUL SYLLABLE PAG..HANGUL SYLLABLE PAH
D329..D343    ; LVT # Lo  [27] HANGUL SYLLABLE PAEG..HANGUL SYLLABLE PAEH
D345..D35F    ; LVT # Lo  [27] HANGUL SYLLABLE PYAG..HANGUL SYLLABLE PYAH
D361..D37B    ; LVT # Lo  [27] HANGUL SYLLABLE PYAEG..HANGUL SYLLABLE PYAEH
D37D..D397    ; LVT # Lo  [27] HANGUL SYLLABLE PEOG..HANGUL SYLLABLE PEOH
D399..D3B3    ; LVT # Lo  [27] HANGUL SYLLABLE PEG..HANGUL SYLLABLE PEH
D3B5..D3CF    ; LVT # Lo  [27] HANGUL SYLLABLE PYEOG..HANGUL SYLLABLE PYEOH
D3D1..D3EB    ; LVT # Lo  [27] HANGUL SYLLABLE PYEG..HANGUL SYLLABLE PYEH
D3ED..D407    ; LVT # Lo  [27] HANGUL SYLLABLE POG..HANGUL SYLLABLE POH
D409..D423    ; LVT # Lo  [27] HANGUL SYLLABLE PWAG..HANGUL SYLLABLE PWAH
D425..D43F    ; LVT # Lo  [27] HANGUL SYLLABLE PWAEG..HANGUL SYLLABLE PWAEH
D441..D45B    ; LVT # Lo  [27] HANGUL SYLLABLE POEG..HANGUL SYLLABLE POEH
D45D..D477    ; LVT # Lo  [27] HANGUL SYLLABLE PYOG..HANGUL SYLLABLE PYOH
D479..D493    ; LVT # Lo  [27] HANGUL SYLLABLE PUG..HANGUL SYLLABLE PUH
D495..D4AF    ; LVT # Lo  [27] HANGUL SYLLABLE PWEOG..HANGUL SYLLABLE PWEOH
D4B1..D4CB    ; LVT # Lo  [27] HANGUL SYLLABLE PWEG..HANGUL SYLLABLE PWEH
D4CD..D4E7    ; LVT # Lo  [27] HANGUL SYLLABLE PWIG..HANGUL SYLLABLE PWIH
D4E9..D503    ; LVT # Lo  [27] HANGUL SYLLABLE PYUG..HANGUL SYLLABLE PYUH
D505..D51F    ; LVT # Lo  [27] HANGUL SYLLABLE PEUG..HANGUL SYLLABLE PEUH
D521..D53B    ; LVT # Lo  [27] HANGUL SYLLABLE PYIG..HANGUL SYLLABLE PYIH
D53D..D557    ; LVT # Lo  [27] HANGUL SYLLABLE PIG..HANGUL SYLLABLE PIH
D559..D573    ; LVT # Lo  [27] HANGUL SYLLABLE HAG..HANGUL SYLLABLE HAH
D575..D58F    ; LVT # Lo  [27] HANGUL SYLLABLE HAEG..HANGUL SYLLABLE HAEH
D591..D5AB    ; LVT # Lo  [27] HANGUL SYLLABLE HYAG..HANGUL SYLLABLE HYAH
D5AD..D5C7    ; LVT # Lo  [27] HANGUL SYLLABLE HYAEG..HANGUL SYLLABLE HYAEH
D5C9..D5E3    ; LVT # Lo  [27] HANGUL SYLLABLE HEOG..HANGUL SYLLABLE HEOH
D5E5..D5FF    ; LVT # Lo  [27] HANGUL SYLLABLE HEG..HANGUL SYLLABLE HEH
D601..D61B    ; LVT # Lo  [27] HANGUL SYLLABLE HYEOG..HANGUL SYLLABLE HYEOH
D61D..D637    ; LVT # Lo  [27] HANGUL SYLLABLE HYEG..HANGUL SYLLABLE HYEH
D639..D653    ; LVT # Lo  [27] HANGUL SYLLABLE HOG..HANGUL SYLLABLE HOH
D655..D66F    ; LVT # Lo  [27] HANGUL SYLLABLE HWAG..HANGUL SYLLABLE HWAH
D671..D68B    ; LVT # Lo  [27] HANGUL SYLLABLE HWAEG..HANGUL SYLLABLE HWAEH
D68D..D6A7    ; LVT # Lo  [27] HANGUL SYLLABLE HOEG..HANGUL SYLLABLE HOEH
D6A9..D6C3    ; LVT # Lo  [27] HANGUL SYLLABLE HYOG..HANGUL SYLLABLE HYOH
D6C5..D6DF    ; LVT # Lo  [27] HANGUL SYLLABLE HUG..HANGUL SYLLABLE HUH
D6E1..D6FB    ; LVT # Lo  [27] HANGUL SYLLABLE HWEOG..HANGUL SYLLABLE HWEOH
D6FD..D717    ; LVT # Lo  [27] HANGUL SYLLABLE HWEG..HANGUL SYLLABLE HWEH
D719..D733    ; LVT # Lo  [27] HANGUL SYLLABLE HWIG..HANGUL SYLLABLE HWIH
D735..D74F    ; LVT # Lo  [27] HANGUL SYLLABLE HYUG..HANGUL SYLLABLE HYUH
D751..D76B    ; LVT # Lo  [27] HANGUL SYLLABLE HEUG..HANGUL SYLLABLE HEUH
D76D..D787    ; LVT # Lo  [27] HANGUL SYLLABLE HYIG..HANGUL SYLLABLE HYIH
D789..D7A3    ; LVT # Lo  [27] HANGUL SYLLABLE HIG..HANGUL SYLLABLE HIH

# Total code points: 10773

# ================================================

200D          ; ZWJ # Cf       ZERO WIDTH JOINER

# Total code points: 1

# EOF
`;


enum SetType
{
    Prepend = 'Prepend',
    Control = 'Control',
    Extend = 'Extend',
    SpacingMark = 'SpacingMark',

    Exceptions = 'Exceptions',
    InvisibleStacker = 'InvisibleStacker',
    Virama = 'Virama',
    Consonants = 'Consonants'
};

const PREPEND_SET = new Set<string>([]);
const CONTROL_SET = new Set<string>([]);
const EXTEND_SET = new Set<string>([]);
const SPACING_MARK_SET = new Set<string>([]);
const EXCEPTIONS_SET = new Set<string>(['\u102B', '\u102C', '\u1038', '\u1062', '\u1063', '\u1064', '\u1067', '\u1068', '\u1069', '\u106A', '\u106B', '\u106C', '\u106D', '\u1083', '\u1087', '\u1088', '\u1089', '\u108A', '\u108B', '\u108C', '\u108F', '\u109A', '\u109B', '\u109C', '\u1A61', '\u1A63', '\u1A64', '\uAA7B', '\uAA7D', '\u11720', '\u11721' ]);
const INVISIBLE_STACKER_SET = new Set<string>(['\u{AAF6}', '\u{1193E}', '\u{11D45}', '\u{11D97}', '\u{1BAB}', '\u{10A3F}', '\u{11A47}', '\u{11A99}', '\u{1039}', '\u{11133}', '\u{17D2}', '\u{1A60}']);
const VIRAMA_SET = new Set<string>(['\u{94D}', '\u{9CD}', '\u{A4D}', '\u{ACD}', '\u{B4D}', '\u{C4D}', '\u{CCD}', '\u{D4D}', '\u{1B44}', '\u{A806}', '\u{A8C4}', '\u{110B9}', '\u{111C0}', '\u{11235}', '\u{1134D}', '\u{11442}', '\u{114C2}', '\u{115BF}', '\u{1163F}', '\u{116B6}', '\u{119E0}', '\u{11839}', '\u{11046}', '\u{11C3F}', '\u{1B44}', '\u{A9C0}', '\u{E001}', '\u{E002}' ]);
const CONSONANTS_SET = new Set<string>([
    // Balinese
    'ᬧ', 'ᬩ', 'ᬢ', 'ᬤ', 'ᬓ', 'ᬕ', 'ᬘ', 'ᬚ', 'ᬲ', 'ᬳ', 'ᬫ', 'ᬦ', 'ᬗ', 'ᬜ', 'ᬯ', 'ᬭ', 'ᬮ', 'ᬬ', 'ᬨ', 'ᬪ', 'ᬝ', 'ᬣ', 'ᬥ', 'ᬖ', 'ᬰ', 'ᬱ', 'ᬡ', 'ᬞ', 'ᬟ', 'ᬠ', 'ᬔ', 'ᬙ', 'ᬛ', 'ᭅ', 'ᭆ', 'ᭇ', 'ᭈ', 'ᭉ', 'ᭊ', 'ᭋ', 'ᬋ', 'ᬌ', 'ᬍ', 'ᬎ',
    // Devanagari
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ळ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॾ', 'ॿ',
    // Sinhala
    'ක', 'ග', 'ඟ', 'ච', 'ජ', 'ට', 'ඩ', 'ණ', 'ඬ', 'ත', 'ද', 'ඳ', 'ප', 'බ', 'ම', 'ඹ', 'ය', 'ර', 'ල', 'ළ', 'ව', 'ස', 'හ', 'ඛ', 'ඝ', 'ඥ', 'ඞ', 'ඡ', 'ඣ', 'ඤ', 'ඦ', 'ඨ', 'ඪ', 'ථ', 'ධ', 'න', 'ඵ', 'භ', 'ශ', 'ෂ', 'ෆ',
    // Javanese
    'ꦧ', 'ꦕ', 'ꦢ', 'ꦝ', 'ꦒ', 'ꦲ', 'ꦗ', 'ꦏ', 'ꦭ', 'ꦩ', 'ꦤ', 'ꦚ', 'ꦔ', 'ꦥ', 'ꦫ', 'ꦱ', 'ꦠ', 'ꦛ', 'ꦮ', 'ꦪ', 'ꦨ', 'ꦖ', 'ꦓ', 'ꦑ', 'ꦟ', 'ꦘ', 'ꦦ', 'ꦬ', 'ꦯ', 'ꦡ', 'ꦣ', 'ꦞ', 'ꦙ', 'ꦰ', 'ꦜ', 'ꦐ', 'ꦉ', 'ꦊ', 'ꦋ',
    // Malayalam
    'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ത', 'ഥ', 'ദ', 'ധ', 'ന', 'ഩ', 'പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ഺ', 'ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ൿ',
    // Bengali
    'প', 'ফ', 'ব', 'ভ', 'ত', 'থ', 'দ', 'ধ', 'ট', 'ঠ', 'ড', 'ঢ', 'ক', 'খ', 'গ', 'ঘ', 'চ', 'ছ', 'জ', 'য', 'ঝ', 'স', 'শ', 'ষ', 'হ', 'ম', 'ন', 'ঙ', 'ঞ', 'ণ', 'ৱ', 'র', 'ৰ', 'ল', 'ৎ',
    // Gurmukhi
    'ਪ', 'ਭ', 'ਬ', 'ਫ', 'ਤ', 'ਧ', 'ਦ', 'ਥ', 'ਚ', 'ਝ', 'ਜ', 'ਛ', 'ਟ', 'ਢ', 'ਡ', 'ਠ', 'ਕ', 'ਘ', 'ਗ', 'ਖ', 'ਵ', 'ਸ', 'ਹ', 'ਮ', 'ਨ', 'ਞ', 'ਣ', 'ਙ', 'ਯ', 'ਰ', 'ੜ', 'ਲ', 'ਫ਼', 'ਜ਼', 'ਸ਼', 'ਖ਼', 'ਗ਼', 'ਲ਼', 'ੲ', 'ੳ',
]);

const SETS: {[K in SetType]: Set<string>} = (() => {
    
    const sets: {[K in SetType]: Set<string>} = {
        [SetType.Prepend]: PREPEND_SET,
        [SetType.Control]: CONTROL_SET,
        [SetType.Extend]: EXTEND_SET,
        [SetType.SpacingMark]: SPACING_MARK_SET,
        [SetType.Exceptions]: EXCEPTIONS_SET,
        [SetType.InvisibleStacker]: INVISIBLE_STACKER_SET,
        [SetType.Virama]: VIRAMA_SET,
        [SetType.Consonants]: CONSONANTS_SET
    };

    const list = RAW_PROPERTY_LIST.split('\n')
    for (let i=0; i < list.length; i++) {
        list[i] = list[i].replace(/\s+/g,' ');
        if (list[i] == '' || list[i].startsWith('#')) { continue; }

        const spaceArray = list[i].split(' ');
        const type = spaceArray[2];
        if (!(type in sets)) { continue;  }

        if (spaceArray[0].includes('..')) {
            const cpoints = spaceArray[0].split('.');
            let start = parseInt(cpoints[0], 16);
            const end = parseInt(cpoints[2], 16);
            for (let j = start; j < end + 1; j++) {
                sets[type as SetType].add(String.fromCodePoint(start));
                start++;
            }
        } else { 
            sets[type as SetType].add(String.fromCodePoint(parseInt(spaceArray[0], 16)));
        }
    }

    // add special substitions that need to be treated as marks to Extend
    sets[SetType.Extend].add('\uE006');
    sets[SetType.Extend].add('\uE007');

    return sets;

})();

// Port of Orthographic Syllables Segmenter from r12a.
// @see https://r12a.github.io/scripts/apps/graphemes/index.html
//
// "Orthographic syllables string together grapheme clusters that should 
// not be broken during edit operations such as hyphenation, letter-spacing, 
// first-letter selection, etc. The app uses home-grown algorithms to handle 
// (so far) Hindi, Bengali, Gurmukhi, Sinhala, Tamil, Malayalam, Balinese, 
// Javanese, and Burmese. It may handle other scripts reasonably well, 
// especially non-Indic ones, but you should check the results.
//
class UnicodeOrthographicSyllablesSegmenter
{
    private makeGraphemeClusters(str: string): string[] {
        let strArray = [...str];
        let out = [];
        let clust = '';
        let prependFlag = false;
        
        for (var i=0; i < strArray.length; i++) {
            if (EXTEND_SET.has(strArray[i])) {
                clust += strArray[i];
            }
            else if (SPACING_MARK_SET.has(strArray[i])) {
                clust += strArray[i]
            }
            else if (CONTROL_SET.has(strArray[i])) {
                clust += strArray[i]
            }
            else if (PREPEND_SET.has(strArray[i])) {
                out.push(clust);
                clust = strArray[i];
                prependFlag = true
            }
            else {
                if (prependFlag) {
                    clust += strArray[i]
                    prependFlag = false
                } else {
                    if (i > 0) { out.push(clust); }
                    clust = strArray[i]
                }
            }
        }
        out.push(clust);
        return out
    }

    segment(inputStr: string)  {    
        let a = this.makeGraphemeClusters(inputStr);
        
        // make temporary tokens
        a = a.map(v => v
            // for Sinhala, V+ZWJ and ZWJ+V, added to Virama
            .replace(/\u0DCA\u200D/g,'\u{E001}')
            .replace(/\u200D\u0DCA/g,'\u{E002}')
            // for Tamil, KSHA and SHRIx2
            .replace(/\u0B95\u0BCD\u007C\u0BB7/g,'\u{E003}')
            .replace(/\u0BB8\u0BCD\u007C\u0BB0\u0BC0/g,'\u{E004}')
            .replace(/\u0BB6\u0BCD\u007C\u0BB0\u0BC0/g,'\u{E005}'));
    
        let out = [];
        let clust = ''
        for (let i = 0; i < a.length - 1; i++) {            
            if ((VIRAMA_SET.has(a[i][a[i].length - 1])
                    && CONSONANTS_SET.has(a[i + 1][0]))
                || INVISIBLE_STACKER_SET.has(a[i][a[i].length - 1])
                || EXCEPTIONS_SET.has(a[i + 1][0])) {
                clust += a[i];
            }
            else {
                clust += a[i];
                out.push(clust);
                clust = '';
            }
        }
        
        clust += a[a.length - 1];
        out.push(clust);
        
        // reinstate 
        out = out.map(v => v
            // Sinhala code points
            .replace(/\uE001/g,'\u0DCA\u200D')
            .replace(/\uE002/g,'\u200D\u0DCA')
            // Tamil conjuncts
            .replace(/\uE003/g,'\u0B95\u0BCD\u0BB7')
            .replace(/\uE004/g,'ஸ்ரீ')
            .replace(/\uE005/g,'ஶ்ரீ'));
    
        return out;
    }

};

export function stringToRenderSegments(inputStr: string): string[] {
    // NOTE: Intl segmenter incorrectly splits by characters,
    //       some of which are wholly diacritics only. Splitting
    //       by some kind of reusable word parts is preferred.
    //
    //return splitTextByGraphemesWithIntl(inputStr);
    const segments = (new UnicodeOrthographicSyllablesSegmenter()).segment(inputStr);
    // console.log(inputStr, segments);
    return segments;
}
