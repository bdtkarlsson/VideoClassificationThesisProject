import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

/**
 * Class containing the network models. Model 1 is a simple CNN model. Model 2 is a AlexNet inspired model. Model 3 is a
 * LRCN inspired model.
 */
public class NetworkModels {

    public static MultiLayerConfiguration getModel1(int video_height, int video_width, int channels, int nrOfOutputs) {

        Random rand = new Random();
        double nonZeroBias = 1;
        double dropOut = 0.5;

        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(rand.nextInt(10000))
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(1)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-3)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .biasLearningRate(1e-2*2)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{10, 10}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(64)
                        .build())

                .layer(1, new SubsamplingLayer.Builder(poolingType, new int[]{4, 4}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{2, 2}, new int[]{2, 2})
                        .name("cnn2")
                        .nOut(128)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(poolingType, new int[]{4, 4}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(2048)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(nrOfOutputs)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .cnnInputSize(video_height, video_width, channels);

        return conf.build();
    }

    static public MultiLayerConfiguration getModel2(int video_height, int video_width, int channels, int nrOfOutputs) {

        double nonZeroBias = 1;
        double dropOut = 0.5;
        Random rand = new Random();

        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(rand.nextInt(10000))
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(1)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder()
                        .name("lrn1")
                        .build())
                .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                        .name("lrn2")
                        .k(2).n(5).alpha(1e-4).beta(0.75)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool3")
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(nrOfOutputs)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .cnnInputSize(video_height, video_width, channels);

        return conf.build();
    }

    public static MultiLayerConfiguration getModel3(int video_height, int video_width, int channels, int nrOfOutputs, int nrOfFrames) {

        double learningRate = 0.001;
        double learningRate2 = 0.001;
        Random rand = new Random();

        //Set up network architecture:
        Updater updater = Updater.RMSPROP;
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(rand.nextInt(10000))
                .regularization(true).l2(0.001) //l2 regularization on all layers
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(learningRate2)
                .updater(updater)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .name("conv1")
                        .nIn(channels) //3 channels: RGB
                        .nOut(32)
                        .kernelSize(14, 14)
                        .stride(7, 7)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .build())   //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool1")
                        .kernelSize(3, 3)
                        .stride(2, 2).build())   //(31-3+0)/2+1 = 15
                .layer(2, new ConvolutionLayer.Builder()
                        .name("conv2")
                        .nIn(32)
                        .nOut(64)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .build())   //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                .layer(3, new DenseLayer.Builder()
                        .name("ffn1")
                        .activation("relu")
                        .nIn(3136)
                        .nOut(256)
                        .weightInit(WeightInit.RELU)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(learningRate)
                        .build())
                .layer(4, new GravesLSTM.Builder()
                        .name("lstm1")
                        .activation("softsign")
                        .nIn(256)
                        .nOut(256)
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(learningRate)
                        .build())
                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .name("output")
                        .activation("softmax")
                        .nIn(256)
                        .nOut(nrOfOutputs)    //4 possible shapes: circle, square, arc, line
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(video_height, video_width, channels))
                .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 64))
                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(nrOfFrames / 5)
                .tBPTTBackwardLength(nrOfFrames / 5);

        return conf.build();

    }

}
