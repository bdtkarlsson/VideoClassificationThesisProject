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
 * Author: Daniel Karlsson c11dkn@cs.umu.se
 *
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
     * @param nrOfCategories The number of possible labels/categories
     * @return The DataSetIterator containing the frames and the corresponding labels
     * @throws IOException
     */
    public static DataSetIterator[] getNonSequentialData(String path, String[] allowedExtensions, int frame_height,
                                                       int frame_width, int channels, int miniBatchSize,
                                                       int percentage, int nrOfCategories) throws IOException {

        /*Read the parent directory containing the data subfolders*/
        File parentDir = new File(path);
        /*Create file split, label maker and path filter*/
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(0), allowedExtensions, labelMaker);
        /*Get the data files, split into two parts*/
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, percentage, 100 - percentage);

        /*Get first part*/
        InputSplit trainingData = filesInDirSplit[0];
        System.out.println("Data 0 size: " + trainingData.length());
        /*Get second part*/
        InputSplit testingData = filesInDirSplit[1];
        System.out.println("Data 1 size: " + testingData.length());

        /*Retrieve data set iterator of first data part*/
        ImageRecordReader reader = new ImageRecordReader(frame_height, frame_width, channels ,labelMaker);
        reader.setLabels(LabelMap.labels); //determine label order
        reader.initialize(trainingData);
        DataSetIterator trainingIter = new RecordReaderDataSetIterator(reader, miniBatchSize, 1, nrOfCategories);

        /*Retrieve data set iterator of second data part*/
        reader = new ImageRecordReader(frame_height, frame_width, channels ,labelMaker);
        reader.setLabels(LabelMap.labels); //determine label order
        reader.initialize(testingData);
        DataSetIterator testingIter = new RecordReaderDataSetIterator(reader, miniBatchSize, 1, nrOfCategories);

        return new DataSetIterator[] {trainingIter, testingIter};
    }

    /**
     * Method for retrieving sequential data. Grabs frames from video clips.
     * The data should be stored as numbered video files with the labels stored in a
     * separate txt file with the same name and number.
     *
     * @param path Path to the folder containing the videos and the labels files
     * @param fileNameStandard The name of the video and label files (e.g. "sportclip_%d")
     * @param startIdx The start index of the video and label files
     * @param nExamples The number of data to be loaded
     * @param miniBatchSize The minibatch size
     * @param startFrame The first frame to be loaded
     * @param nrFrames The number of frames to be loaded from each video file
     * @param video_height The height of the video
     * @param video_width The width of the video
     * @param nrOfCategories The number of possible labels/categories
     * @return The DataSetIterator containing the frames and the corresponding labels
     * @throws Exception
     */
    public static DataSetIterator getSequentialData(String path, String fileNameStandard, int startIdx,
                                                    int nExamples, int miniBatchSize, int startFrame,  int nrFrames,
                                                    int video_height, int video_width, int nrOfCategories) throws Exception {

        /*Get the features*/
        SequenceRecordReader featuresTrain = getFeaturesReader(path + "/" + fileNameStandard + ".mp4", startIdx, nExamples, startFrame,
                nrFrames, video_height, video_width);
        /*Get the labels*/
        SequenceRecordReader labelsTrain = getLabelsReader(path + "/" + fileNameStandard + ".txt", startIdx, nExamples);
        /*Create a Data set iterator with the features and the labels*/
        SequenceRecordReaderDataSetIterator sequenceIter =
                new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, nrOfCategories, false);
        sequenceIter.setPreProcessor(new VideoPreProcessor());

        /*AsyncDataSetIterator: Used to (pre-load) load data in a separate thread*/
        return new AsyncDataSetIterator(sequenceIter, 1);
    }

    /**
     *
     * @param fullPath The path to the video with the features
     * @param startIdx The ID of the start video
     * @param numOfFiles The number of files
     * @param startFrame The ID of the start frame
     * @param nrFrames The number of frames
     * @param video_height The height of the video
     * @param video_width The width of the video
     * @return Record Reader with the features
     * @throws Exception
     */
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

    /**
     *
     * @param fullPath The path to the file with the categories
     * @param startIdx The ID of the start category file
     * @param numOfFiles The number of files
     * @return RecordReader with the categories
     * @throws Exception
     */
    private static SequenceRecordReader getLabelsReader(String fullPath, int startIdx, int numOfFiles) throws Exception {
        /*Get the files containing the labels*/
        InputSplit isLabels = new NumberedFileInputSplit(fullPath, startIdx, startIdx + numOfFiles - 1);
        /*Get the labels*/
        CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);
        return csvSeq;
    }

    /**
     * PreProcessor for the features. Scales the values from 0-255 to 0-1
     */
    private static class VideoPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            toPreProcess.getFeatureMatrix().divi(255);  /*[0,255] -> [0,1] for input pixel values*/
        }
    }

}
