/**
 * Converts the loop's input parts into the Interactions API wire format before they reach
 * `callModel`. The API rejects unknown fields on parts (e.g. `media_resolution`), so image and
 * video parts keep only type, mime_type, resolution and data/uri.
 */
export function toWireParts(input, defaultResolution = 'high') {
  if (!Array.isArray(input)) return input;
  return input.map((item) => {
    if (!item || typeof item !== 'object') return { type: 'text', text: String(item) };
    if (item.type === 'text') return { type: 'text', text: String(item.text ?? '') };
    if (item.type === 'image' || item.type === 'video') {
      const resolution = item.resolution || item.media_resolution || item.mediaResolution || defaultResolution;
      const out = {
        type: item.type,
        mime_type: item.mime_type || item.mimeType || (item.type === 'image' ? 'image/png' : 'video/webm'),
        resolution,
      };
      if (item.data) out.data = item.data;
      if (item.uri) out.uri = item.uri;
      return out;
    }
    return item;
  });
}
