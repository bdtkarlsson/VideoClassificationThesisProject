import org.datavec.api.conf.Configuration;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Class which contains methods for loading video frame data in DataSetIterator form. The sequential data is read
 * directly from video files and uses the JCodecs FrameGrab8Bit class to extract the frames. The non-sequential data
 * is read from frames directly (stored as image files).
 */
public class DataLoader {


    /**
     * Method for retrieving non-sequential data. The data should be stored as images with each label having
     * its own subfolder.
     * @param path Path to the folder containing the label subfolders which contain the frame images
     * @param allowedExtensions The allowed extensions of the images (e.g. bmp, jpg etc.)
     * @param frame_height  The height of the frame
     * @param frame_width The width of the frame
     * @param channels Number of channels (e.g. 3 for RGB images)
     * @param miniBatchSize The minibatch size
     * @param percentage The percentage of the images in the path that will be loaded
     * @param upper True if the upper percentage of the images should be loaded and false if the lower percentage
     *              should be loaded
     * @param nrOfPossibleLabels The number of possible labels/categories
     * @return The DataSetIterator containing the frames and the corresponding labels
     * @throws IOException
     */
    public static DataSetIterator getNonSequentialData(String path, String[] allowedExtensions, int frame_height,
                                                       int frame_width, int channels, int miniBatchSize,
                                                       int percentage, boolean upper, int nrOfPossibleLabels) throws IOException {

        File parentDir = new File(path);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, new Random(100));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(100), allowedExtensions, labelMaker);

        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, percentage, 100 - percentage);
        int i = 0;
        if(!upper) {
            i = 1;
        }
        InputSplit data = filesInDirSplit[i];

        ImageRecordReader reader = new ImageRecordReader(frame_height, frame_width, channels ,labelMaker);
        reader.initialize(data);

        DataSetIterator iter = new RecordReaderDataSetIterator(reader, miniBatchSize, 0, nrOfPossibleLabels);

        return iter;
    }

    /**
     * Method for retrieving sequential data. Grabs frames from video clips.
     * The data should be stored as numbered video files with the labels stored in a
     * separate txt file with the same name and number.
     *
     * @param path Path to the folder containing the videos and the labels files
     * @param fileName The name of the video and label files
     * @param extension The extension of the video files
     * @param startIdx The start index of the video and label files
     * @param nExamples The number of data to be loaded
     * @param miniBatchSize The minibatch size
     * @param startFrame The first frame to be loaded
     * @param nrFrames The number of frames to be loaded from each video file
     * @param video_height The height of the video
     * @param video_width The width of the video
     * @param nrOfPossibleLabels The number of possible labels/categories
     * @return The DataSetIterator containing the frames and the corresponding labels
     * @throws Exception
     */
    public static DataSetIterator getSequentialData(String path, String fileName, String extension, int startIdx,
                                                    int nExamples, int miniBatchSize, int startFrame,  int nrFrames,
                                                    int video_height, int video_width, int nrOfPossibleLabels) throws Exception {
        /*Create the full path to the filenames*/
        String fullPathVideo = path + fileName + "%d." + extension;
        String fullPathLabel = path + fileName + "%d.txt";
        /*Get the features*/
        SequenceRecordReader featuresTrain = getFeaturesReader(fullPathVideo, startIdx, nExamples, startFrame,
                nrFrames, video_height, video_width);
        /*Get the labels*/
        SequenceRecordReader labelsTrain = getLabelsReader(fullPathLabel, startIdx, nExamples);
        /*Create a Data set iterator with the features and the labels*/
        SequenceRecordReaderDataSetIterator sequenceIter =
                new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, nrOfPossibleLabels, false);
        sequenceIter.setPreProcessor(new VideoPreProcessor());

        /*AsyncDataSetIterator: Used to (pre-load) load data in a separate thread*/
        return new AsyncDataSetIterator(sequenceIter, 1);
    }

    private static SequenceRecordReader getFeaturesReader(String fullPath, int startIdx, int numOfFiles, int startFrame,
                                                          int nrFrames, int video_height, int video_width) throws Exception {
        /*Get the files containing the video files*/
        InputSplit is = new NumberedFileInputSplit(fullPath, startIdx, startIdx + numOfFiles - 1);
        /*Set up configuration*/
        Configuration conf = new Configuration();
        conf.set(SequentialFramesRecordReader.RAVEL, "true");
        conf.set(SequentialFramesRecordReader.START_FRAME, "" + startFrame);
        conf.set(SequentialFramesRecordReader.TOTAL_FRAMES, String.valueOf(nrFrames));
        conf.set(SequentialFramesRecordReader.ROWS, String.valueOf(video_width));
        conf.set(SequentialFramesRecordReader.COLUMNS, String.valueOf(video_height));
        /*Get the features*/
        SequentialFramesRecordReader crr = new SequentialFramesRecordReader();
        crr.initialize(conf, is);
        return crr;
    }

    private static SequenceRecordReader getLabelsReader(String fullPath, int startIdx, int num) throws Exception {
        /*Get the files containing the labels*/
        InputSplit isLabels = new NumberedFileInputSplit(fullPath, startIdx, startIdx + num - 1);
        /*Get the labels*/
        CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);
        return csvSeq;
    }

    private static class VideoPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            toPreProcess.getFeatureMatrix().divi(255);  /*[0,255] -> [0,1] for input pixel values*/
        }
    }

}