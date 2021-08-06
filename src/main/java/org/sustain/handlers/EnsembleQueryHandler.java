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
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.sustain.Collection;
import org.sustain.*;
import org.sustain.modeling.*;
import org.sustain.util.Constants;
import org.sustain.util.CountyClusters;
import org.sustain.util.FancyLogger;
import scala.Tuple2;
import scala.collection.Iterator;

import java.io.File;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.Future;

public class EnsembleQueryHandler extends GrpcSparkHandler<ModelRequest, ModelResponse> {

    private String model_save_path = "/s/chopin/e/proj/sustain/sapmitra/spark_mongo/saved_models/";
    private static final double targetVariance = 0.95;
    private static final double samplingPercentage = 0.0003;
    private static final Logger log = LogManager.getLogger(EnsembleQueryHandler.class);
    private String filename = "driver.txt";

    public static void main(String arg[]) {

        Map<String, List> opMap = CountyClusters.extractCountyGroups("./src/main/java/org/sustain/handlers/clusters.csv");
        System.out.println(opMap);
    }

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
        private Map<String, RandomForestRegressionModel> trained_parents;
        private Map<String, String> reverseChildToParentMap = null;
        private Map<String, Double> parentRmse = null;

        RFRegressionTask(ModelRequest modelRequest, List<String> gisJoins, Map<String, Double> parentRmse, Map<String, RandomForestRegressionModel> trained_parents) {
            this.rfRequest = modelRequest.getRForestRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
            this.trained_parents = trained_parents;
            this.parentRmse = parentRmse;
        }

        RFRegressionTask(ModelRequest modelRequest, List<String> gisJoins, Map<String, RandomForestRegressionModel> trained_parents,
                         Map<String, Double> parentRmse, Map<String, String> reverseChildToParentMap) {
            this.rfRequest = modelRequest.getRForestRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
            this.trained_parents = trained_parents;
            this.parentRmse = parentRmse;
            this.reverseChildToParentMap = reverseChildToParentMap;
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

            List<ModelResponse> modelResponses = new ArrayList<>();

            for (String gisJoin : this.gisJoins) {

                // TRAINING PARENTS
                if (reverseChildToParentMap == null) {
                    RFRegressionExhaustiveModel model = new RFRegressionExhaustiveModel(mongoUri, dbName, collection.getName(),
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

                        synchronized (this) {
                            trained_parents.put(gisJoin, model.getTrained_rfModel());
                            parentRmse.put(gisJoin,model.getRmse());
                        }

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
                } else {
                    // TRAINING CHILDREN
                    RFChildIncrementalModel model = new RFChildIncrementalModel(mongoUri, dbName, collection.getName(),
                            gisJoin);

                    String parentGisJoin = reverseChildToParentMap.get(gisJoin);
                    RandomForestRegressionModel parentRegressionModel = trained_parents.get(parentGisJoin);
                    double targetRMSE = parentRmse.get(parentGisJoin);
                    model.setParent_rfModel(parentRegressionModel);
                    model.setParent_rmse(targetRMSE);
                    model.parentGisJoin = parentGisJoin;
                    model.setFilename();

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
            }
            return modelResponses;
        }
    }


    protected class GBRegressionTask implements SparkTask<List<ModelResponse>> {
        private final GBoostRegressionRequest gbRequest;
        private final Collection requestCollection;
        private final List<String> gisJoins;
        private Map<String, GBTRegressionModel> trained_parents;
        private Map<String, String> reverseChildToParentMap = null;
        private Map<String, Double> parentRmse = null;

        GBRegressionTask(ModelRequest modelRequest, List<String> gisJoins, Map<String, Double> parentRmse, Map<String, GBTRegressionModel> trained_parents) {
            this.gbRequest = modelRequest.getGBoostRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
            this.trained_parents = trained_parents;
            this.parentRmse = parentRmse;
        }


        GBRegressionTask(ModelRequest modelRequest, List<String> gisJoins, Map<String, GBTRegressionModel> trained_parents,
                         Map<String, Double> parentRmse, Map<String, String> reverseChildToParentMap) {
            this.gbRequest = modelRequest.getGBoostRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
            this.trained_parents = trained_parents;
            this.reverseChildToParentMap = reverseChildToParentMap;
            this.parentRmse = parentRmse;
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

            List<ModelResponse> modelResponses = new ArrayList<>();

            for (String gisJoin : this.gisJoins) {

                // TRAINING PARENTS
                if (reverseChildToParentMap == null) {
                    //GBoostRegressionModel model = new GBoostRegressionModel(mongoUri, dbName, collection.getName(), gisJoin);
                    GBoostRegressionExhaustiveModel model = new GBoostRegressionExhaustiveModel(mongoUri, dbName, collection.getName(), gisJoin);

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
                        synchronized (this) {
                            trained_parents.put(gisJoin, model.getTrained_gbModel());
                            parentRmse.put(gisJoin,model.getRmse());
                        }

                        GBoostRegressionResponse rsp = GBoostRegressionResponse.newBuilder()
                                .setGisJoin(model.getGisJoin())
                                .setRmse(model.getRmse())
                                .setR2(model.getR2())
                                .build();

                        modelResponses.add(ModelResponse.newBuilder()
                                .setGBoostRegressionResponse(rsp)
                                .build());
                    } else {
                        log.info("Ran into a problem building a model for GISJoin {}, skipping.", gisJoin);
                    }
                } else {
                    // TRAINING CHILDREN

                    GBoostChildIncrementalModel model = new GBoostChildIncrementalModel(mongoUri, dbName, collection.getName(), gisJoin);

                    String parentGisJoin = reverseChildToParentMap.get(gisJoin);
                    GBTRegressionModel parentRegressionModel = trained_parents.get(parentGisJoin);
                    double targetRMSE = parentRmse.get(parentGisJoin);
                    model.setParent_gbModel(parentRegressionModel);
                    model.setParent_rmse(targetRMSE);
                    model.parentGisJoin = parentGisJoin;
                    model.setFilename();

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

                        GBoostRegressionResponse rsp = GBoostRegressionResponse.newBuilder()
                                .setGisJoin(model.getGisJoin())
                                .setRmse(model.getRmse())
                                .setR2(model.getR2())
                                .build();

                        modelResponses.add(ModelResponse.newBuilder()
                                .setGBoostRegressionResponse(rsp)
                                .build());
                    } else {
                        log.info("Ran into a problem building a model for GISJoin {}, skipping.", gisJoin);
                    }

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


    private boolean find_mode(ModelRequest request) {
        int val = 7;
        if (request.getType().equals(ModelType.R_FOREST_REGRESSION)){
            RForestRegressionRequest req = request.getRForestRegressionRequest();
            val = req.getMaxDepth();

        } else if (request.getType().equals(ModelType.G_BOOST_REGRESSION)) {
            GBoostRegressionRequest req = request.getGBoostRegressionRequest();
            val = req.getMaxDepth();

        }

        if (val >=0) {
            System.out.println("EXHAUSTIVE MODE !!!!!!!!!!!!!!!!!!!!!!!!!!!");
            return true;
        } else {
            System.out.println("TL MODE !!!!!!!!!!!!!!!!!!!!!!!!!!!");
            return false;
        }

    }

    @Override
    public void handleRequest() {

        String full_log_string = "";
        if (isValid(this.request)) {

            boolean isExhaustive = find_mode(this.request);

            // PARSE THE CLUSTER CSV
            Map<String, List> clusterCSVMap = CountyClusters.extractCountyGroups("./src/main/java/org/sustain/handlers/clusters_test.csv");
            full_log_string+=FancyLogger.fancy_logging("CLUSTER_MAP: "+clusterCSVMap, null);
            if (request.getType().equals(ModelType.R_FOREST_REGRESSION)) {
                {

                    try {
                        RForestRegressionRequest req = this.request.getRForestRegressionRequest();

                        Map<String,String> reverseChildToParentMap = new HashMap<>();

                        // IGNORE GISJOIN SENT BY THE REQUEST. INSTEAD GET THE FIRST ENTRY
                        java.util.Iterator<Map.Entry<String, List>> iterator = clusterCSVMap.entrySet().iterator();

                        List<String> parents = new ArrayList<>();

                        // A COMBINED LIST OF ALL DEPENDENT GISJOINS
                        List<String> allchildrenGISList = new ArrayList<>();

                        // MAP CONTAINING ALL PARENT MODELS
                        Map<String, RandomForestRegressionModel> trained_parents_map = new HashMap<>();
                        Map<String, Double> parentRMSEMap = new HashMap<>();

                        while (iterator.hasNext()) {
                            Map.Entry<String, List> firstEntry = iterator.next();
                            String parentGisJoin = firstEntry.getKey();
                            parents.add(parentGisJoin);
                            List<String> childrenGisJoin = firstEntry.getValue();
                            allchildrenGISList.addAll(childrenGisJoin);

                            for(String child: childrenGisJoin) {
                                reverseChildToParentMap.put(child, parentGisJoin);
                            }
                        }

                        List<List<String>> gisJoinBatches_parent = batchGisJoins(parents, 20);

                        if(isExhaustive) {
                            // EXHAUSTIVELY TRAINING ALL THE PARENTS FIRST

                            // ****************START PARENT TRAINING ***********************
                            List<Future<List<ModelResponse>>> batchedModelTasks_parents = new ArrayList<>();
                            for (List<String> gisJoinBatch : gisJoinBatches_parent) {
                                RFRegressionTask gbTask = new RFRegressionTask(this.request, gisJoinBatch, parentRMSEMap, trained_parents_map);
                                batchedModelTasks_parents.add(this.sparkManager.submit(gbTask, "gb-regression-query"));
                            }

                            // Wait for each task to complete and return their ModelResponses
                            for (Future<List<ModelResponse>> indvTask : batchedModelTasks_parents) {
                                List<ModelResponse> batchedModelResponses = indvTask.get();
                                for (ModelResponse modelResponse : batchedModelResponses) {
                                    //DON'T THINK WE NEED RESPONSE OBSERVER HERE....NOTHING TO PASS BACK. JUST OBSERVE THE RESULTS
                                    full_log_string += FancyLogger.fancy_logging("RECEIVED A RESPONSE FOR " + modelResponse.getRForestRegressionResponse().getGisJoin(), log);
                                    this.responseObserver.onNext(modelResponse);

                                }
                            }

                            // JUST ITERATING AND PRINTING THE TRAINED MODELS
                            java.util.Iterator<Map.Entry<String, RandomForestRegressionModel>> iterator_trained_models = trained_parents_map.entrySet().iterator();
                            int cnt = 0;
                            while (iterator_trained_models.hasNext()) {
                                Map.Entry<String, RandomForestRegressionModel> firstEntry = iterator_trained_models.next();
                                String parentGisJoin = firstEntry.getKey();
                                //parents.add(parentGisJoin);
                                RandomForestRegressionModel trained_model = firstEntry.getValue();

                                trained_model.save(model_save_path+parentGisJoin+".model");
                                // LOG SOMETHING TO CHECK TRAINED MODEL IS HERE
                                full_log_string += FancyLogger.fancy_logging("Model Num " + (cnt + 1) + "GISJOIN: " +
                                        parentGisJoin + " Trained Model: " + trained_model.paramMap(), null);
                                cnt++;
                            }

                            full_log_string += FancyLogger.fancy_logging("PARENT TRAINING CONCLUDED!!!!!", null);
                            System.out.println("PARENT TRAINING CONCLUDED!!!!!");
                            System.out.println(full_log_string);
                            FancyLogger.write_out(full_log_string, filename);

                        } else {
                            // ****************END PARENT TRAINING ***********************
                            full_log_string += FancyLogger.fancy_logging("CHILDREN TRAINING STARTING,.....", null);
                            System.out.println("CHILDREN TRAINING STARTING,.....");
                            for(String parent: parents) {
                                File tempFile = new File(model_save_path+parent+".model");
                                boolean isSaved = tempFile.exists();
                                if (isSaved) {
                                    RandomForestRegressionModel model = RandomForestRegressionModel.load(model_save_path + parent + ".model");
                                    trained_parents_map.put(parent, model);
                                } else{
                                    System.out.println("SAVED MODEL NOT FOUND FOR:"+ parent);
                                }
                            }

                            // ****************START CHILDREN TRAINING ***********************

                            // BATCHING ALL THE CHILDREN INTO GROUPS OF 20
                            List<List<String>> gisJoinBatches_children = batchGisJoins(allchildrenGISList, 20);

                            List<Future<List<ModelResponse>>> batchedModelTasks_children = new ArrayList<>();
                            for (List<String> gisJoinBatch : gisJoinBatches_children) {

                                RFRegressionTask rfTask = new RFRegressionTask(this.request, gisJoinBatch, trained_parents_map, parentRMSEMap, reverseChildToParentMap);
                                batchedModelTasks_children.add(this.sparkManager.submit(rfTask, "gb-regression-query"));
                            }

                            // Wait for each task to complete and return their ModelResponses
                            for (Future<List<ModelResponse>> indvTask : batchedModelTasks_children) {
                                List<ModelResponse> batchedModelResponses = indvTask.get();
                                for (ModelResponse modelResponse : batchedModelResponses) {
                                    //DON'T THINK WE NEED RESPONSE OBSERVER HERE....NOTHING TO PASS BACK. JUST OBSERVE THE RESULTS
                                    full_log_string += FancyLogger.fancy_logging("RECEIVED A RESPONSE FOR " + modelResponse.getRForestRegressionResponse().getGisJoin(), log);
                                    this.responseObserver.onNext(modelResponse);

                                }
                            }

                            System.out.println(full_log_string);
                            FancyLogger.write_out(full_log_string, filename);

                        }

                    } catch (Exception e) {
                        log.error("Failed to evaluate query", e);
                        responseObserver.onError(e);
                    }
                }
            } else if(request.getType().equals(ModelType.G_BOOST_REGRESSION)) {

                try {
                    GBoostRegressionRequest req = this.request.getGBoostRegressionRequest();

                    Map<String,String> reverseChildToParentMap = new HashMap<>();

                    // IGNORE GISJOIN SENT BY THE REQUEST. INSTEAD GET THE FIRST ENTRY
                    java.util.Iterator<Map.Entry<String, List>> iterator = clusterCSVMap.entrySet().iterator();

                    List<String> parents = new ArrayList<>();

                    // A COMBINED LIST OF ALL DEPENDENT GISJOINS
                    List<String> allchildrenGISList = new ArrayList<>();

                    // MAP CONTAINING ALL PARENT MODELS
                    Map<String, GBTRegressionModel> trained_parents_map = new HashMap<>();
                    Map<String, Double> parentRMSEMap = new HashMap<>();

                    while (iterator.hasNext()) {
                        Map.Entry<String, List> firstEntry = iterator.next();
                        String parentGisJoin = firstEntry.getKey();
                        parents.add(parentGisJoin);
                        List<String> childrenGisJoin = firstEntry.getValue();
                        allchildrenGISList.addAll(childrenGisJoin);

                        for(String child: childrenGisJoin) {
                            reverseChildToParentMap.put(child, parentGisJoin);
                        }
                    }

                    List<List<String>> gisJoinBatches_parent = batchGisJoins(parents, 20);

                    if(isExhaustive) {
                        // EXHAUSTIVELY TRAINING ALL THE PARENTS FIRST

                        // ****************START PARENT TRAINING ***********************
                        List<Future<List<ModelResponse>>> batchedModelTasks_parents = new ArrayList<>();
                        for (List<String> gisJoinBatch : gisJoinBatches_parent) {
                            GBRegressionTask gbTask = new GBRegressionTask(this.request, gisJoinBatch, parentRMSEMap, trained_parents_map);
                            batchedModelTasks_parents.add(this.sparkManager.submit(gbTask, "gb-regression-query"));
                        }

                        // Wait for each task to complete and return their ModelResponses
                        for (Future<List<ModelResponse>> indvTask : batchedModelTasks_parents) {
                            List<ModelResponse> batchedModelResponses = indvTask.get();
                            for (ModelResponse modelResponse : batchedModelResponses) {
                                //DON'T THINK WE NEED RESPONSE OBSERVER HERE....NOTHING TO PASS BACK. JUST OBSERVE THE RESULTS
                                full_log_string += FancyLogger.fancy_logging("RECEIVED A RESPONSE FOR " + modelResponse.getGBoostRegressionResponse().getGisJoin(), log);
                                this.responseObserver.onNext(modelResponse);

                            }
                        }

                        // JUST ITERATING AND PRINTING THE TRAINED MODELS
                        java.util.Iterator<Map.Entry<String, GBTRegressionModel>> iterator_trained_models = trained_parents_map.entrySet().iterator();
                        int cnt = 0;
                        while (iterator_trained_models.hasNext()) {
                            Map.Entry<String, GBTRegressionModel> firstEntry = iterator_trained_models.next();
                            String parentGisJoin = firstEntry.getKey();
                            parents.add(parentGisJoin);
                            GBTRegressionModel trained_model = firstEntry.getValue();
                            trained_model.save(model_save_path+parentGisJoin+".model");
                            // LOG SOMETHING TO CHECK TRAINED MODEL IS HERE
                            full_log_string += FancyLogger.fancy_logging("Model Num " + (cnt + 1) + "GISJOIN: " +
                                    parentGisJoin + " Trained Model: " + trained_model.paramMap(), null);
                            cnt++;
                        }

                        full_log_string += FancyLogger.fancy_logging("PARENT TRAINING CONCLUDED!!!!!", null);
                        System.out.println("PARENT TRAINING CONCLUDED!!!!!");

                        System.out.println(full_log_string);
                        FancyLogger.write_out(full_log_string, filename);

                    } else {
                        // ****************END PARENT TRAINING ***********************
                        full_log_string += FancyLogger.fancy_logging("CHILDREN TRAINING STARTING,.....", null);
                        System.out.println("CHILDREN TRAINING STARTING,.....");

                        for(String parent: parents) {
                            File tempFile = new File(model_save_path+parent+".model");
                            boolean isSaved = tempFile.exists();
                            if (isSaved) {
                                GBTRegressionModel model = GBTRegressionModel.load(model_save_path + parent + ".model");
                                trained_parents_map.put(parent, model);
                            } else{
                                System.out.println("SAVED MODEL NOT FOUND FOR:"+ parent);
                            }
                        }

                        // ****************START CHILDREN TRAINING ***********************

                        // BATCHING ALL THE CHILDREN INTO GROUPS OF 20
                        List<List<String>> gisJoinBatches_children = batchGisJoins(allchildrenGISList, 20);

                        List<Future<List<ModelResponse>>> batchedModelTasks_children = new ArrayList<>();
                        for (List<String> gisJoinBatch : gisJoinBatches_children) {

                            GBRegressionTask gbTask = new GBRegressionTask(this.request, gisJoinBatch, trained_parents_map, parentRMSEMap, reverseChildToParentMap);
                            batchedModelTasks_children.add(this.sparkManager.submit(gbTask, "gb-regression-query"));
                        }

                        // Wait for each task to complete and return their ModelResponses
                        for (Future<List<ModelResponse>> indvTask : batchedModelTasks_children) {
                            List<ModelResponse> batchedModelResponses = indvTask.get();
                            for (ModelResponse modelResponse : batchedModelResponses) {
                                //DON'T THINK WE NEED RESPONSE OBSERVER HERE....NOTHING TO PASS BACK. JUST OBSERVE THE RESULTS
                                full_log_string += FancyLogger.fancy_logging("RECEIVED A RESPONSE FOR " + modelResponse.getGBoostRegressionResponse().getGisJoin(), log);
                                this.responseObserver.onNext(modelResponse);

                            }
                        }

                        System.out.println(full_log_string);
                        FancyLogger.write_out(full_log_string, filename);
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
