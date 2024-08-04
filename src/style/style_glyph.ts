import type {AlphaImage} from '../util/image';

/**
 * Some metices related to a glyph
 */
export type GlyphMetrics = {
    width: number;
    height: number;
    left: number;
    top: number;
    advance: number;
    /**
     * isDoubleResolution = true for 48px textures
     */
    isDoubleResolution?: boolean;
};

/**
 * A style glyph type
 */
export type StyleGlyph = {
    id: string;
    bitmap: AlphaImage;
    metrics: GlyphMetrics;
};
