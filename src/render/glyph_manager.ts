import {loadGlyphRange} from '../style/load_glyph_range';

import {unicodeBlockLookup} from '../util/is_char_in_unicode_block';
import {AlphaImage} from '../util/image';

import type {StyleGlyph} from '../style/style_glyph';
import type {RequestManager} from '../util/request_manager';
import type {GetGlyphsResponse} from '../util/actor_messages';

type Entry = {
    // null means we've requested the range, but the glyph wasn't included in the result.
    glyphs: {
        [id: number]: StyleGlyph | null;
    };
    requests: {
        [range: number]: Promise<{[_: number]: StyleGlyph | null}>;
    };
    ranges: {
        [range: number]: boolean | null;
    };
    tinySDF1?: TinySDF;
    tinySDF2?: TinySDF;
};

const INF = 1e20;

// Mapbox TinySDF modified to use TextMetric.fontBoundBoxAdvance
// when rendering word segments. 
//
// In our case, multi-lingual and complex script labels are most likely in one font 
// hence measuing bbox of a segment without knowing baseline of surrounding segments
// leads to segments visually jumping up and down in a line. 
// 
// @see util.ts, UnicodeOrthographicSyllablesSegmenter
//
// @see https://github.com/mapbox/tiny-sdf/blob/main/index.js
//
class TinySDF {
    buffer: number;
    cutoff: number;
    radius: number;
    size: number;
    ctx: CanvasRenderingContext2D;
    gridOuter: Float64Array;
    gridInner: Float64Array;
    f: Float64Array;
    z: Float64Array;
    v: Uint16Array;
    constructor({
        fontSize = 24,
        buffer = 3,
        radius = 8,
        cutoff = 0.25,
        fontFamily = 'sans-serif',
        fontWeight = 'normal',
        fontStyle = 'normal'
    } = {}) {
        this.buffer = buffer;
        this.cutoff = cutoff;
        this.radius = radius;

        // make the canvas size big enough to both have the specified buffer around the glyph
        // for "halo", and account for some glyphs possibly being larger than their font size
        const size = this.size = fontSize + buffer * 8;

        const canvas = this._createCanvas(size);
        const ctx = this.ctx = canvas.getContext('2d', {willReadFrequently: true});
        ctx.font = `${fontStyle} ${fontWeight} ${fontSize}px ${fontFamily}`;

        ctx.textBaseline = 'alphabetic';
        ctx.textAlign = 'left'; // Necessary so that RTL text doesn't have different alignment
        ctx.fillStyle = 'black';

        // TODO: experiment without anti-aliasing

        // temporary arrays for the distance transform
        this.gridOuter = new Float64Array(size * size);
        this.gridInner = new Float64Array(size * size);
        this.f = new Float64Array(size);
        this.z = new Float64Array(size + 1);
        this.v = new Uint16Array(size);
    }

    _createCanvas(size) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = size;
        return canvas;
    }

    draw(char: string) {
        const {
            width: glyphAdvance,
            fontBoundingBoxAscent,
            fontBoundingBoxDescent,
            actualBoundingBoxLeft,
            actualBoundingBoxRight
        } = this.ctx.measureText(char);

        // The integer/pixel part of the top alignment is encoded in metrics.glyphTop
        // The remainder is implicitly encoded in the rasterization
        const bboxVerticalExpandK = 2; // expand bbox vertically for hanging accents
        const glyphTop = Math.ceil(fontBoundingBoxAscent * 1.2); // 1.2 and 1.8 are random number to accomodate Myanmar
        const glyphLeft = 0;

        // If the glyph overflows the canvas size, it will be clipped at the bottom/right
        const glyphWidth = Math.max(0, Math.min(this.size - this.buffer, Math.ceil(actualBoundingBoxRight - actualBoundingBoxLeft)));
        const glyphHeight = Math.min(this.size - this.buffer, glyphTop + Math.ceil(fontBoundingBoxDescent * 1.8));

        const width = glyphWidth + 2 * this.buffer;
        const height = glyphHeight + 2 * this.buffer;

        const len = Math.max(width * height, 0);
        const data = new Uint8ClampedArray(len);
        const glyph = {data, width, height, glyphWidth, glyphHeight, glyphTop, glyphLeft, glyphAdvance};
        if (glyphWidth === 0 || glyphHeight === 0) return glyph;

        const {ctx, buffer, gridInner, gridOuter} = this;
        ctx.clearRect(buffer, buffer, glyphWidth, glyphHeight);
        ctx.fillText(char, buffer, buffer + glyphTop);
        const imgData = ctx.getImageData(buffer, buffer, glyphWidth, glyphHeight);

        // Initialize grids outside the glyph range to alpha 0
        gridOuter.fill(INF, 0, len);
        gridInner.fill(0, 0, len);

        for (let y = 0; y < glyphHeight; y++) {
            for (let x = 0; x < glyphWidth; x++) {
                const a = imgData.data[4 * (y * glyphWidth + x) + 3] / 255; // alpha value
                if (a === 0) continue; // empty pixels

                const j = (y + buffer) * width + x + buffer;

                if (a === 1) { // fully drawn pixels
                    gridOuter[j] = 0;
                    gridInner[j] = INF;

                } else { // aliased pixels
                    const d = 0.5 - a;
                    gridOuter[j] = d > 0 ? d * d : 0;
                    gridInner[j] = d < 0 ? d * d : 0;
                }
            }
        }

        edt(gridOuter, 0, 0, width, height, width, this.f, this.v, this.z);
        edt(gridInner, buffer, buffer, glyphWidth, glyphHeight, width, this.f, this.v, this.z);

        for (let i = 0; i < len; i++) {
            const d = Math.sqrt(gridOuter[i]) - Math.sqrt(gridInner[i]);
            data[i] = Math.round(255 - 255 * (d / this.radius + this.cutoff));
        }

        return glyph;
    }
}

// 2D Euclidean squared distance transform by Felzenszwalb & Huttenlocher https://cs.brown.edu/~pff/papers/dt-final.pdf
function edt(data: Float64Array, x0: number, y0: number, width: number, height: number, gridSize: number, f: Float64Array, v: Uint16Array, z: Float64Array) {
    for (let x = x0; x < x0 + width; x++) edt1d(data, y0 * gridSize + x, gridSize, height, f, v, z);
    for (let y = y0; y < y0 + height; y++) edt1d(data, y * gridSize + x0, 1, width, f, v, z);
}

// 1D squared distance transform
function edt1d(grid: Float64Array, offset: number, stride: number, length: number, f: Float64Array, v: Uint16Array, z: Float64Array) {
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    f[0] = grid[offset];

    for (let q = 1, k = 0, s = 0; q < length; q++) {
        f[q] = grid[offset + q * stride];
        const q2 = q * q;
        do {
            const r = v[k];
            s = (f[q] - f[r] + q2 - r * r) / (q - r) / 2;
        } while (s <= z[k] && --k > -1);

        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = INF;
    }

    for (let q = 0, k = 0; q < length; q++) {
        while (z[k + 1] < q) k++;
        const r = v[k];
        const qr = q - r;
        grid[offset + q * stride] = f[r] + qr * qr;
    }
}

export class GlyphManager {
    requestManager: RequestManager;
    localIdeographFontFamily: string | false;
    entries: {[stack: string]: Entry};
    url: string;

    // exposed as statics to enable stubbing in unit tests
    static loadGlyphRange = loadGlyphRange;
    static TinySDF = TinySDF;

    constructor(requestManager: RequestManager, localIdeographFontFamily?: string | false) {
        this.requestManager = requestManager;
        this.localIdeographFontFamily = localIdeographFontFamily;
        this.entries = {};
    }

    setURL(url?: string | null) {
        this.url = url;
    }

    async getGlyphs(glyphs: {[stack: string]: Array<string>}): Promise<GetGlyphsResponse> {
        const glyphsPromises: Promise<{stack: string; id: string; glyph: StyleGlyph}>[] = [];

        for (const stack in glyphs) {
            for (const id of glyphs[stack]) {
                glyphsPromises.push(this._getAndCacheGlyphsPromise(stack, id));
            }
        }

        const updatedGlyphs = await Promise.all(glyphsPromises);

        const result: GetGlyphsResponse = {};

        for (const {stack, id, glyph} of updatedGlyphs) {
            if (!result[stack]) {
                result[stack] = {};
            }
            // Clone the glyph so that our own copy of its ArrayBuffer doesn't get transferred.
            result[stack][id] = glyph && {
                id: glyph.id,
                bitmap: glyph.bitmap.clone(),
                metrics: glyph.metrics
            };
        }

        return result;
    }

    async _getAndCacheGlyphsPromise(stack: string, id: string): Promise<{stack: string; id: string; glyph: StyleGlyph}> {
        let entry = this.entries[stack];
        if (!entry) {
            entry = this.entries[stack] = {
                glyphs: {},
                requests: {},
                ranges: {}
            };
        }

        let glyph = entry.glyphs[id];

        if (id.trim() == '') {
            return {stack, id, glyph};
        }

        // glyphs for which we override pbf fonts with tinysdf have priority,
        // because codepage is not aligned with font, e.g. one pbf font can have > 1 codepage
        if (this._doesSegmentSupportLocalGlyph(id)) {
            const glyph = this._tinySDF(entry, stack, id);
            if (glyph) {
                entry.glyphs[id] = glyph;
                return {stack, id, glyph};
            }
        }

        if (glyph !== undefined) {
            return {stack, id, glyph};
        }
        
        const codePoint = id.codePointAt(0);
        // non-printable
        if (typeof(codePoint) === 'undefined') {
            return {stack, id, glyph: null};
        }

        const range = Math.floor(codePoint / 256);
        if (range * 256 > 65535) {
            throw new Error('glyphs > 65535 not supported');
        }

        if (entry.ranges[range]) {
            return {stack, id, glyph};
        }

        if (!this.url) {
            throw new Error('glyphsUrl is not set');
        }

        if (!entry.requests[range]) {
            const promise = GlyphManager.loadGlyphRange(stack, range, this.url, this.requestManager);
            entry.requests[range] = promise;
        }

        const response = await entry.requests[range];
        for (const id in response) {
            entry.glyphs[id] = response[id];
        }
        entry.ranges[range] = true;
        return {stack, id, glyph: response[id] || null};
    }

    _doesSegmentSupportLocalGlyph(id: string): boolean {
        /* eslint-disable new-cap */
        const code = id.codePointAt(0);
        return !!this.localIdeographFontFamily &&
            (unicodeBlockLookup['Devanagari'](code) ||
            unicodeBlockLookup['Bengali'](code) ||
            unicodeBlockLookup['Gujarati'](code) ||
            unicodeBlockLookup['Tamil'](code) ||
            unicodeBlockLookup['Telugu'](code) ||
            unicodeBlockLookup['Tibetan'](code) ||
            unicodeBlockLookup['Myanmar'](code) ||
            unicodeBlockLookup['Khmer'](code) ||
            unicodeBlockLookup['CJK Unified Ideographs'](code) ||
            unicodeBlockLookup['Hangul Syllables'](code) ||
            unicodeBlockLookup['Hiragana'](code) ||
            unicodeBlockLookup['Katakana'](code));
        /* eslint-enable new-cap */
    }

    _doesCharShouldHaveDoubleResolution(id: string): boolean {
        /* eslint-disable new-cap */
        const code = id.codePointAt(0);
        return unicodeBlockLookup['CJK Unified Ideographs'](code) ||
            unicodeBlockLookup['Hangul Syllables'](code) ||
            unicodeBlockLookup['Hiragana'](code) ||
            unicodeBlockLookup['Katakana'](code);
        /* eslint-enable new-cap */
    }    

    _tinySDF(entry: Entry, stack: string, id: string): StyleGlyph {
        const fontFamily = this.localIdeographFontFamily;
        if (!fontFamily) {
            return;
        }

        // Client-generated glyphs are rendered at 2x texture scale,
        // because CJK glyphs are more detailed than others.
        const isDoubleResolution = this._doesCharShouldHaveDoubleResolution(id);
        const textureScale = isDoubleResolution ? 2 : 1;

        let tinySDF = isDoubleResolution ? entry.tinySDF2 : entry.tinySDF1;
        if (!tinySDF) {
            let fontWeight = '400'; 
            if (/bold/i.test(stack)) {
                fontWeight = '900';
            } else if (/medium/i.test(stack)) {
                fontWeight = '500';
            } else if (/light/i.test(stack)) {
                fontWeight = '200';
            }
            tinySDF = new GlyphManager.TinySDF({
                fontSize: 24 * textureScale,
                buffer: 8 * textureScale,
                radius: 8 * textureScale,
                cutoff: 0.25,
                fontFamily,
                fontWeight
            });

            if (isDoubleResolution) {
                entry.tinySDF2 = tinySDF;
            } else {
                entry.tinySDF1 = tinySDF;
            }
        }

        const char = tinySDF.draw(id as any);

        /**
         * TinySDF's "top" is the distance from the alphabetic baseline to the top of the glyph.
         * Server-generated fonts specify "top" relative to an origin above the em box (the origin
         * comes from FreeType, but I'm unclear on exactly how it's derived)
         * ref: https://github.com/mapbox/sdf-glyph-foundry
         *
         * Server fonts don't yet include baseline information, so we can't line up exactly with them
         * (and they don't line up with each other)
         * ref: https://github.com/mapbox/node-fontnik/pull/160
         *
         * To approximately align TinySDF glyphs with server-provided glyphs, we use this baseline adjustment
         * factor calibrated to be in between DIN Pro and Arial Unicode (but closer to Arial Unicode)
         */
        const topAdjustment = 27.5;

        const leftAdjustment = 0.5;

        return {
            id,
            bitmap: new AlphaImage({width: char.width || 30 * textureScale, height: char.height || 30 * textureScale}, char.data),
            metrics: {
                width: char.glyphWidth / textureScale || 24,
                height: char.glyphHeight / textureScale || 24,
                left: (char.glyphLeft / textureScale + leftAdjustment) || 0,
                top: char.glyphTop / textureScale - topAdjustment || -8,
                advance: char.glyphAdvance / textureScale || 24,
                isDoubleResolution: isDoubleResolution
            }
        };
    }
}
