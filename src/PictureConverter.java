import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture8Bit;
import org.jcodec.scale.ColorUtil;
import org.jcodec.scale.RgbToBgr8Bit;
import org.jcodec.scale.Transform8Bit;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/**
 * Created by bdtkarlsson on 2016-10-27.
 */
public class PictureConverter {

    public static BufferedImage toBufferedImage8Bit(Picture8Bit src) {
        if (src.getColor() != ColorSpace.RGB) {
            Transform8Bit transform = ColorUtil.getTransform8Bit(src.getColor(), ColorSpace.RGB);
            Picture8Bit rgb = Picture8Bit.createCropped(src.getWidth(), src.getHeight(), ColorSpace.RGB, src.getCrop());
            transform.transform(src, rgb);
            new RgbToBgr8Bit().transform(rgb, rgb);
            src = rgb;
        }

        BufferedImage dst = new BufferedImage(src.getCroppedWidth(), src.getCroppedHeight(),
                BufferedImage.TYPE_3BYTE_BGR);

        if (src.getCrop() == null)
            toBufferedImage8Bit(src, dst);
        else
            toBufferedImageCropped8Bit(src, dst);

        return dst;
    }

    private static void toBufferedImageCropped8Bit(Picture8Bit src, BufferedImage dst) {
        byte[] data = ((DataBufferByte) dst.getRaster().getDataBuffer()).getData();
        byte[] srcData = src.getPlaneData(0);
        int dstStride = dst.getWidth() * 3;
        int srcStride = src.getWidth() * 3;
        for (int line = 0, srcOff = 0, dstOff = 0; line < dst.getHeight(); line++) {
            for (int id = dstOff, is = srcOff; id < dstOff + dstStride; id += 3, is += 3) {
                // Unshifting, since JCodec stores [0..255] -> [-128, 127]
                data[id] = (byte) (srcData[is] + 128);
                data[id + 1] = (byte) (srcData[is + 1] + 128);
                data[id + 2] = (byte) (srcData[is + 2] + 128);
            }
            srcOff += srcStride;
            dstOff += dstStride;
        }
    }

    public static void toBufferedImage8Bit(Picture8Bit src, BufferedImage dst) {
        byte[] data = ((DataBufferByte) dst.getRaster().getDataBuffer()).getData();
        byte[] srcData = src.getPlaneData(0);
        for (int i = 0; i < data.length; i++) {
            // Unshifting, since JCodec stores [0..255] -> [-128, 127]
            data[i] = (byte) (srcData[i] + 128);
        }
    }


}
