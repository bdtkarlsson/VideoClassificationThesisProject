import org.apache.commons.compress.utils.IOUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.loader.ImageLoader;
import org.jcodec.api.FrameGrab8Bit;
import org.jcodec.api.JCodecException;
import org.jcodec.common.io.ByteBufferSeekableByteChannel;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.common.io.SeekableByteChannel;
import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture8Bit;
import org.jcodec.scale.ColorUtil;
import org.jcodec.scale.RgbToBgr8Bit;
import org.jcodec.scale.Transform8Bit;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 *  *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 * !NOTE!
 *
 * This is an updated version of the CodecRecordReader provided by DeepLearning4j copyrighted by Skymind, Inc. The original CodecRecordReader
 * used an old version of Jcodecs (1.5) which resulted in error when grabbing some frames. The new version (2.0)
 * depreceates the FrameGrab class and instead uses the FrameGrab8Bit class which is more stable and does not generate
 * error for some frames.
 */
public class SequentialFramesRecordReader extends FileRecordReader implements SequenceRecordReader {
    private int startFrame = 0;
    private int numFrames = -1;
    private int totalFrames = -1;
    private double framesPerSecond = -1.0D;
    private double videoLength = -1.0D;
    private ImageLoader imageLoader;
    private boolean ravel = false;
    public static final String NAME_SPACE = "org.datavec.codec.reader";
    public static final String ROWS = "org.datavec.codec.reader.rows";
    public static final String COLUMNS = "org.datavec.codec.reader.columns";
    public static final String START_FRAME = "org.datavec.codec.reader.startframe";
    public static final String TOTAL_FRAMES = "org.datavec.codec.reader.frames";
    public static final String TIME_SLICE = "org.datavec.codec.reader.time";
    public static final String RAVEL = "org.datavec.codec.reader.ravel";
    public static final String VIDEO_DURATION = "org.datavec.codec.reader.duration";

    public SequentialFramesRecordReader() {
    }

    public List<List<Writable>> sequenceRecord() {
        File next = (File)this.iter.next();

        try {
            return this.loadData(NIOUtils.readableChannel(next), next);
        } catch (IOException var3) {
            throw new RuntimeException(var3);
        }
    }

    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        byte[] data = IOUtils.toByteArray(dataInputStream);
        ByteBuffer bb = ByteBuffer.wrap(data);
        SequentialFramesRecordReader.FixedByteBufferSeekableByteChannel sbc = new SequentialFramesRecordReader.FixedByteBufferSeekableByteChannel(bb);
        return this.loadData(sbc, null);
    }

    private List<List<Writable>> loadData(SeekableByteChannel seekableByteChannel, File f) throws IOException {
        ArrayList record = new ArrayList();
        Picture8Bit p = null;
        BufferedImage e;
        if(this.numFrames >= 1) {
            FrameGrab8Bit i;
            try {
                i = FrameGrab8Bit.createFrameGrab8Bit(seekableByteChannel);
                if(this.startFrame != 0) {
                    i.seekToFramePrecise(this.startFrame);
                }
            } catch (JCodecException var8) {
                System.err.println("1");
                throw new RuntimeException(var8);
            }

            for(int i1 = this.startFrame; i1 < this.startFrame + this.numFrames; ++i1) {
                try {
                    p = i.getNativeFrame();
                    if(p == null) {
                        p = FrameGrab8Bit.getFrameFromFile(f, i1);
                    }
                    e = toBufferedImage8Bit(p);
                    if(this.ravel) {
                        record.add(RecordConverter.toRecord(this.imageLoader.toRaveledTensor(e)));
                    } else {
                        record.add(RecordConverter.toRecord(this.imageLoader.asRowVector(e)));
                    }
                } catch (Exception var7) {
                    try {
                        p = FrameGrab8Bit.getFrameFromFile(f, i1);
                    } catch (JCodecException e1) {
                        e1.printStackTrace();
                    }
                    e = toBufferedImage8Bit(p);
                }
            }
        } else {
            if(this.framesPerSecond < 1.0D) {
                throw new IllegalStateException("No frames or frame time intervals specified");
            }

            for(double var9 = 0.0D; var9 < this.videoLength; var9 += this.framesPerSecond) {
                try {
                    p = FrameGrab8Bit.getFrameFromChannelAtSec(seekableByteChannel, var9);
                    e = toBufferedImage8Bit(p);
                    if(this.ravel) {
                        record.add(RecordConverter.toRecord(this.imageLoader.toRaveledTensor(e)));
                    } else {
                        record.add(RecordConverter.toRecord(this.imageLoader.asRowVector(e)));
                    }
                } catch (Exception var6) {
                    System.err.println("3");
                    throw new RuntimeException(var6);
                }
            }
        }

        return record;
    }

    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.setConf(conf);
        this.initialize(split);
    }

    public List<Writable> next() {
        throw new UnsupportedOperationException("next() not supported for CodecRecordReader (use: sequenceRecord)");
    }

    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("record(URI,DataInputStream) not supported for CodecRecordReader");
    }

    public boolean hasNext() {
        return this.iter.hasNext();
    }

    public void setConf(Configuration conf) {
        super.setConf(conf);
        this.startFrame = conf.getInt("org.datavec.codec.reader.startframe", 0);
        this.numFrames = conf.getInt("org.datavec.codec.reader.frames", -1);
        int rows = conf.getInt("org.datavec.codec.reader.rows", 28);
        int cols = conf.getInt("org.datavec.codec.reader.columns", 28);
        this.imageLoader = new ImageLoader(rows, cols);
        this.framesPerSecond = (double)conf.getFloat("org.datavec.codec.reader.time", -1.0F);
        this.videoLength = (double)conf.getFloat("org.datavec.codec.reader.duration", -1.0F);
        this.ravel = conf.getBoolean("org.datavec.codec.reader.ravel", false);
        this.totalFrames = conf.getInt("org.datavec.codec.reader.frames", -1);
    }

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

    public Configuration getConf() {
        return super.getConf();
    }

    private static class FixedByteBufferSeekableByteChannel extends ByteBufferSeekableByteChannel {
        private ByteBuffer backing;

        public FixedByteBufferSeekableByteChannel(ByteBuffer backing) {
            super(backing);

            try {
                Field e = this.getClass().getSuperclass().getDeclaredField("maxPos");
                e.setAccessible(true);
                e.set(this, Integer.valueOf(backing.limit()));
            } catch (Exception var3) {
                throw new RuntimeException(var3);
            }

            this.backing = backing;
        }

        public int read(ByteBuffer dst) throws IOException {
            return !this.backing.hasRemaining()?-1:super.read(dst);
        }
    }
}
