/* ========================================================
 * EnsembleQueryHandler.java
 *   Captures input parameters into out regression model object
 *
 * Author: Saptashwa Mitra
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * ======================================================== */
package org.sustain.handlers;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.sustain.*;
import org.sustain.Collection;
import org.sustain.SparkTask;
import org.sustain.modeling.GBoostRegressionModel;
import org.sustain.modeling.RFRegressionModel;
import org.sustain.util.Constants;
import org.sustain.SparkManager;
import org.sustain.util.FancyLogger;
import scala.Tuple2;
import scala.collection.Iterator;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.util.*;
import java.util.concurrent.Future;

public class EnsembleQueryHandler extends GrpcSparkHandler<ModelRequest, ModelResponse> {

    private static final double targetVariance = 0.95;
    private static final double samplingPercentage = 0.0003;
    private static final Logger log = LogManager.getLogger(EnsembleQueryHandler.class);

    public EnsembleQueryHandler(ModelRequest request, StreamObserver<ModelResponse> responseObserver, SparkManager sparkManager) {
        super(request, responseObserver, sparkManager);
    }

    public int getNoOfPrincipalComponentsByVariance(PCAModel pca, double targetVariance) {
        int n;
        double varianceSum = 0.0;
        DenseVector explainedVariance = pca.explainedVariance();
        Iterator<Tuple2<Object, Object>> iterator = explainedVariance.iterator();
        while (iterator.hasNext()) {
            Tuple2<Object, Object> next = iterator.next();
            n = Integer.parseInt(next._1().toString()) + 1;
            if (n >= pca.getK()) {
                break;
            }
            varianceSum += Double.parseDouble(next._2().toString());
            if (varianceSum >= targetVariance) {
                return n;
            }
        }

        return pca.getK();
    }

    protected class RFRegressionTask implements SparkTask<List<ModelResponse>> {
        private final RForestRegressionRequest rfRequest;
        private final Collection requestCollection;
        private final List<String> gisJoins;

        RFRegressionTask(ModelRequest modelRequest, List<String> gisJoins) {
            this.rfRequest = modelRequest.getRForestRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
        }

        @Override
        public List<ModelResponse> execute(JavaSparkContext sparkContext) throws Exception {

            String mongoUri = String.format("mongodb://%s:%d", Constants.DB.HOST, Constants.DB.PORT);
            String dbName = Constants.DB.NAME;

            Collection collection = requestCollection; // We only support 1 collection currently

            // Initailize ReadConfig
            Map<String, String> readOverrides = new HashMap<String, String>();
            readOverrides.put("spark.mongodb.input.collection", requestCollection.getName());
            readOverrides.put("spark.mongodb.input.database", Constants.DB.NAME);
            readOverrides.put("spark.mongodb.input.uri", mongoUri);

            ReadConfig readConfig = ReadConfig.create(sparkContext.getConf(), readOverrides);

            // FETCHING MONGO COLLECTION ONCE FOR ALL MODELS
            Dataset<Row> mongocollection = MongoSpark.load(sparkContext, readConfig).toDF();

            // THE FEATURES THAT NEED COMPRESSING
            // WORK WITH ALL FEATURES FOR NOW
            List<String> allFeaturesList = new ArrayList<String>();

            String colnInfo = "";

            for(Tuple2<String,String> dataType: mongocollection.dtypes()) {
                colnInfo+= dataType._1+" "+dataType._2+"\n";
                if (!(dataType._2.toLowerCase().contains("string"))) {
                    allFeaturesList.add(dataType._1);
                }
            }

            // THE FEATURES THAT NEED COMPRESSING
            // WORK WITH ALL FEATURES FOR NOW
            String[] allFeatures = allFeaturesList.toArray(new String[0]);
            FancyLogger.fancy_logging("ALL FEATURES: "+Arrays.asList(allFeatures)+"\n"+colnInfo, log);
            //*****************PCA START************************************

            // Dropping rows with null values
            Dataset<Row> selectedFeatures = mongocollection.sample(samplingPercentage);
            selectedFeatures = selectedFeatures.na().drop();

            FancyLogger.fancy_logging("FINISHED SAMPLING: "+Arrays.asList(allFeatures), log);

            // Assembling
            VectorAssembler assembler = new VectorAssembler().setInputCols(allFeatures).setOutputCol("features");
            Dataset<Row> featureDF = assembler.transform(selectedFeatures);
            featureDF.show(10);

            // Scaling
            log.info("Normalizing features");
            StandardScaler scaler = new StandardScaler()
                    .setInputCol("features")
                    .setOutputCol("normalized_features");
            StandardScalerModel scalerModel = scaler.fit(featureDF);

            featureDF = scalerModel.transform(featureDF);
            featureDF = featureDF.drop("features");
            featureDF = featureDF.withColumnRenamed("normalized_features", "features");

            log.info("Dataframe after normalizing with StandardScaler");
            //featureDF.show(10);

            //featureDF = featureDF.sample(samplingPercentage);

            // PCA
            long pcaTime1 = System.currentTimeMillis();
            PCAModel pca = new PCA()
                    .setInputCol("features")
                    .setOutputCol("pcaFeatures")
                    .setK(7)
                    .fit(featureDF);

            Dataset<Row> pcaDF = pca.transform(featureDF).select("features", "pcaFeatures");
            long pcaTime2 = System.currentTimeMillis();

            //pcaDF.write().json("sustain-pcaDF.json");
            int requiredNoOfPCs = getNoOfPrincipalComponentsByVariance(pca, 0.95);
            log.info("requiredNoOfPCs: {}", requiredNoOfPCs);

            pcaDF.show(10);
            //*****************PCA STOP*************************************

            if(allFeatures.length > 0) {
                FancyLogger.fancy_logging("PCA COMPLETE", log);
                return null;
            }

            List<ModelResponse> modelResponses = new ArrayList<>();

            for (String gisJoin : this.gisJoins) {
                RFRegressionModel model = new RFRegressionModel(mongoUri, dbName, collection.getName(),
                        gisJoin);

                model.setMongoCollection(mongocollection);
                // Set parameters of Random Forest Regression Model

                int featuresCount = collection.getFeaturesCount();
                String[] features = new String[featuresCount];
                for (int i = 0; i < featuresCount; i++) {
                    features[i] = collection.getFeatures(i);
                }

                model.setFeatures(features);
                model.setLabel(collection.getLabel());
                model.setBootstrap(rfRequest.getIsBootstrap());

                // CHECKING FOR VALID MODEL PARAMETER VALUES
                if (rfRequest.getSubsamplingRate() > 0 && rfRequest.getSubsamplingRate() <= 1)
                    model.setSubsamplingRate(rfRequest.getSubsamplingRate());
                if (rfRequest.getNumTrees() > 0)
                    model.setNumTrees(rfRequest.getNumTrees());
                if (rfRequest.getFeatureSubsetStrategy() != null && !rfRequest.getFeatureSubsetStrategy().isEmpty())
                    model.setFeatureSubsetStrategy(rfRequest.getFeatureSubsetStrategy());
                if (rfRequest.getImpurity() != null && !rfRequest.getImpurity().isEmpty())
                    model.setImpurity(rfRequest.getImpurity());
                if (rfRequest.getMaxDepth() > 0)
                    model.setMaxDepth(rfRequest.getMaxDepth());
                if (rfRequest.getMaxBins() > 0)
                    model.setMaxBins(rfRequest.getMaxBins());
                if (rfRequest.getTrainSplit() > 0 && rfRequest.getTrainSplit() < 1)
                    model.setTrainSplit(rfRequest.getTrainSplit());
                if (rfRequest.getMinInfoGain() > 0)
                    model.setMinInfoGain(rfRequest.getMinInfoGain());
                if (rfRequest.getMinInstancesPerNode() >= 1)
                    model.setMinInstancesPerNode(rfRequest.getMinInstancesPerNode());
                if (rfRequest.getMinWeightFractionPerNode() >= 0.0 && rfRequest.getMinWeightFractionPerNode() < 0.5)
                    model.setMinWeightFractionPerNode(rfRequest.getMinWeightFractionPerNode());


                // Submit task to Spark Manager
                boolean ok = model.train();

                if (ok) {

                    RForestRegressionResponse rsp = RForestRegressionResponse.newBuilder()
                            .setGisJoin(model.getGisJoin())
                            .setRmse(model.getRmse())
                            .setR2(model.getR2())
                            .build();

                    modelResponses.add(ModelResponse.newBuilder()
                            .setRForestRegressionResponse(rsp)
                            .build());
                } else {
                    log.info("Ran into a problem building a model for GISJoin {}, skipping.", gisJoin);
                }
            }
            return modelResponses;
        }
    }


    protected class GBRegressionTask implements SparkTask<List<ModelResponse>> {
        private final GBoostRegressionRequest gbRequest;
        private final Collection requestCollection;
        private final List<String> gisJoins;

        GBRegressionTask(ModelRequest modelRequest, List<String> gisJoins) {
            this.gbRequest = modelRequest.getGBoostRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
        }

        @Override
        public List<ModelResponse> execute(JavaSparkContext sparkContext) throws Exception {

            String mongoUri = String.format("mongodb://%s:%d", Constants.DB.HOST, Constants.DB.PORT);
            String dbName = Constants.DB.NAME;

            Collection collection = requestCollection; // We only support 1 collection currently

            // Initailize ReadConfig
            Map<String, String> readOverrides = new HashMap<String, String>();
            readOverrides.put("spark.mongodb.input.collection", requestCollection.getName());
            readOverrides.put("spark.mongodb.input.database", Constants.DB.NAME);
            readOverrides.put("spark.mongodb.input.uri", mongoUri);

            ReadConfig readConfig = ReadConfig.create(sparkContext.getConf(), readOverrides);

            // FETCHING MONGO COLLECTION ONCE FOR ALL MODELS
            Dataset<Row> mongocollection = MongoSpark.load(sparkContext, readConfig).toDF();

            // THE FEATURES THAT NEED COMPRESSING
            // WORK WITH ALL FEATURES FOR NOW
            /*List<String> allFeaturesList = new ArrayList<String>();



            String colnInfo = "";

            for(Tuple2<String,String> dataType: mongocollection.dtypes()) {
                colnInfo+= dataType._1+" "+dataType._2+"\n";
                if (!(dataType._2.toLowerCase().contains("string"))) {
                    allFeaturesList.add(dataType._1);
                }
            }
            String[] allFeatures = allFeaturesList.toArray(new String[0]);*/

            String[] allFeatures = {"max_eastward_wind","max_max_air_temperature","max_min_air_temperature","max_northward_wind",
                    "max_precipitation","max_specific_humidity","max_surface_downwelling_shortwave_flux_in_air","max_vpd","min_eastward_wind",
                    "min_max_air_temperature","min_min_air_temperature","min_northward_wind","min_precipitation","min_specific_humidity",
                    "min_surface_downwelling_shortwave_flux_in_air","min_vpd","timestamp"};

            FancyLogger.fancy_logging("ALL FEATURES: "+Arrays.asList(allFeatures), log);
            //*****************PCA START************************************

            // Dropping rows with null values
            Dataset<Row> selectedFeatures = mongocollection.sample(samplingPercentage);
            selectedFeatures = selectedFeatures.na().drop();

            FancyLogger.fancy_logging("FINISHED SAMPLING: "+Arrays.asList(allFeatures), log);
            // Assembling
            VectorAssembler assembler = new VectorAssembler().setInputCols(allFeatures).setOutputCol("features");
            Dataset<Row> featureDF = assembler.transform(selectedFeatures);
            //featureDF.show(10);

            // Scaling
            log.info("Normalizing features");
            StandardScaler scaler = new StandardScaler()
                    .setInputCol("features")
                    .setOutputCol("normalized_features");
            StandardScalerModel scalerModel = scaler.fit(featureDF);

            featureDF = scalerModel.transform(featureDF);
            featureDF = featureDF.drop("features");
            featureDF = featureDF.withColumnRenamed("normalized_features", "features");

            log.info("Dataframe after normalizing with StandardScaler");
            //featureDF.show(10);

            featureDF = featureDF.sample(samplingPercentage);

            // PCA
            long pcaTime1 = System.currentTimeMillis();

            FancyLogger.fancy_logging("STARTING PCA",log);
            PCAModel pca = new PCA()
                    .setInputCol("features")
                    .setOutputCol("pcaFeatures")
                    .setK(7)
                    .fit(featureDF);

            Dataset<Row> pcaDF = pca.transform(featureDF).select("features", "pcaFeatures");
            //pcaDF.write().json("sustain-pcaDF.json");
            FancyLogger.fancy_logging("FINISHED PCA",log);
            int requiredNoOfPCs = getNoOfPrincipalComponentsByVariance(pca, 0.95);
            log.info("requiredNoOfPCs: {}", requiredNoOfPCs);

            FancyLogger.fancy_logging("EXTRACTED PCs",log);
            pcaDF.show(10);
            //*****************PCA STOP*************************************

            if(allFeatures.length > 0) {
                FancyLogger.fancy_logging("PCA COMPLETE", log);
                return null;
            }


            List<ModelResponse> modelResponses = new ArrayList<>();

            for (String gisJoin : this.gisJoins) {
                GBoostRegressionModel model = new GBoostRegressionModel(mongoUri, dbName, collection.getName(),
                        gisJoin);

                model.setMongoCollection(mongocollection);
                // Set parameters of Random Forest Regression Model

                int featuresCount = collection.getFeaturesCount();
                String[] features = new String[featuresCount];
                for (int i = 0; i < featuresCount; i++) {
                    features[i] = collection.getFeatures(i);
                }

                model.setFeatures(features);
                model.setLabel(collection.getLabel());

                if (gbRequest.getLossType() != null && !gbRequest.getLossType().isEmpty())
                    model.setLossType(gbRequest.getLossType());
                if (gbRequest.getMaxIter() > 0)
                    model.setMaxIter(gbRequest.getMaxIter());
                if (gbRequest.getSubsamplingRate() > 0 && gbRequest.getSubsamplingRate() <= 1)
                    model.setSubsamplingRate(gbRequest.getSubsamplingRate());
                if (gbRequest.getStepSize() > 0 && gbRequest.getStepSize() <= 1)
                    model.setStepSize(gbRequest.getStepSize());
                if (gbRequest.getFeatureSubsetStrategy() != null && !gbRequest.getFeatureSubsetStrategy().isEmpty())
                    model.setFeatureSubsetStrategy(gbRequest.getFeatureSubsetStrategy());
                if (gbRequest.getImpurity() != null && !gbRequest.getImpurity().isEmpty())
                    model.setImpurity(gbRequest.getImpurity());
                if (gbRequest.getMaxDepth() > 0)
                    model.setMaxDepth(gbRequest.getMaxDepth());
                if (gbRequest.getMaxBins() > 0)
                    model.setMaxBins(gbRequest.getMaxBins());
                if (gbRequest.getTrainSplit() > 0)
                    model.setTrainSplit(gbRequest.getTrainSplit());
                if (gbRequest.getMinInfoGain() > 0)
                    model.setMinInfoGain(gbRequest.getMinInfoGain());
                if (gbRequest.getMinInstancesPerNode() >= 1)
                    model.setMinInstancesPerNode(gbRequest.getMinInstancesPerNode());
                if (gbRequest.getMinWeightFractionPerNode() >= 0.0 && gbRequest.getMinWeightFractionPerNode() < 0.5)
                    model.setMinWeightFractionPerNode(gbRequest.getMinWeightFractionPerNode());


                // Submit task to Spark Manager
                boolean ok = model.train();

                if (ok) {

                    RForestRegressionResponse rsp = RForestRegressionResponse.newBuilder()
                            .setGisJoin(model.getGisJoin())
                            .setRmse(model.getRmse())
                            .setR2(model.getR2())
                            .build();

                    modelResponses.add(ModelResponse.newBuilder()
                            .setRForestRegressionResponse(rsp)
                            .build());
                } else {
                    log.info("Ran into a problem building a model for GISJoin {}, skipping.", gisJoin);
                }
            }
            return modelResponses;
        }
    }

    /**
     * Checks the validity of a ModelRequest object, in the context of a Random Forest Regression request.
     * @param modelRequest The ModelRequest object populated by the gRPC endpoint.
     * @return Boolean true if the model request is valid, false otherwise.
     */
    @Override
    public boolean isValid(ModelRequest modelRequest) {
        if (modelRequest.getType().equals(ModelType.R_FOREST_REGRESSION) || modelRequest.getType().equals(ModelType.G_BOOST_REGRESSION)) {
            if (modelRequest.getCollectionsCount() == 1) {
                if (modelRequest.getCollections(0).getFeaturesCount() > 0) {
                    return (modelRequest.hasRForestRegressionRequest() || modelRequest.hasGBoostRegressionRequest());
                }
            }
        }

        return false;
    }


    private List<List<String>> batchGisJoins(List<String> gisJoins, int batchSize) {
        List<List<String>> batches = new ArrayList<>();
        int totalGisJoins = gisJoins.size();
        int gisJoinsPerBatch = (int) Math.ceil( (1.0*totalGisJoins) / (1.0*batchSize) );
        log.info(">>> Max batch size: {}, totalGisJoins: {}, gisJoinsPerBatch: {}", batchSize, totalGisJoins,
                gisJoinsPerBatch);

        for (int i = 0; i < totalGisJoins; i++) {
            if ( i % gisJoinsPerBatch == 0 ) {
                batches.add(new ArrayList<>());
            }
            String gisJoin = gisJoins.get(i);
            batches.get(batches.size() - 1).add(gisJoin);
        }

        StringBuilder batchLog = new StringBuilder(
                String.format(">>> %d batches for %d GISJoins\n", batches.size(), totalGisJoins)
        );
        for (int i = 0; i < batches.size(); i++) {
            batchLog.append(String.format("\tBatch %d size: %d\n", i, batches.get(i).size()));
        }
        log.info(batchLog.toString());
        return batches;
    }

    @Override
    public void handleRequest() {
        if (isValid(this.request)) {

            if (request.getType().equals(ModelType.R_FOREST_REGRESSION)) {
                try {
                    RForestRegressionRequest req = this.request.getRForestRegressionRequest();

                    List<List<String>> gisJoinBatches = batchGisJoins(req.getGisJoinsList(), 20);

                    List<Future<List<ModelResponse>>> batchedModelTasks = new ArrayList<>();
                    for (List<String> gisJoinBatch: gisJoinBatches) {
                        RFRegressionTask rfTask = new RFRegressionTask(this.request, gisJoinBatch);
                        batchedModelTasks.add(this.sparkManager.submit(rfTask, "rf-regression-query"));
                    }

                    // Wait for each task to complete and return their ModelResponses
                    for (Future<List<ModelResponse>> indvTask: batchedModelTasks) {
                        List<ModelResponse> batchedModelResponses = indvTask.get();
                        for (ModelResponse modelResponse: batchedModelResponses) {
                            this.responseObserver.onNext(modelResponse);
                        }
                    }

                } catch (Exception e) {
                    log.error("Failed to evaluate query", e);
                    responseObserver.onError(e);
                }
            } else if(request.getType().equals(ModelType.G_BOOST_REGRESSION)) {

                try {
                    GBoostRegressionRequest req = this.request.getGBoostRegressionRequest();

                    List<List<String>> gisJoinBatches = batchGisJoins(req.getGisJoinsList(), 20);

                    List<Future<List<ModelResponse>>> batchedModelTasks = new ArrayList<>();
                    for (List<String> gisJoinBatch: gisJoinBatches) {
                        GBRegressionTask gbTask = new GBRegressionTask(this.request, gisJoinBatch);
                        batchedModelTasks.add(this.sparkManager.submit(gbTask, "gb-regression-query"));
                    }

                    // Wait for each task to complete and return their ModelResponses
                    for (Future<List<ModelResponse>> indvTask: batchedModelTasks) {
                        List<ModelResponse> batchedModelResponses = indvTask.get();
                        for (ModelResponse modelResponse: batchedModelResponses) {
                            this.responseObserver.onNext(modelResponse);
                        }
                    }

                } catch (Exception e) {
                    log.error("Failed to evaluate query", e);
                    responseObserver.onError(e);
                }
            }
        } else {
            log.warn("Invalid Model Request!");
        }
    }
}
