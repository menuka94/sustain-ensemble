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
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.sustain.Collection;
import org.sustain.*;
import org.sustain.modeling.*;
import org.sustain.util.Constants;
import org.sustain.util.CountyClusters;
import org.sustain.util.FancyLogger;

import java.util.*;
import java.util.concurrent.Future;


// THIS IS NOT REALLY AN ENSEMBLE HANDLER....THIS IS CHECKING DISTRIBUTION SIMILARITY
public class EnsembleQueryHandler_unnecessary extends GrpcSparkHandler<ModelRequest, ModelResponse> {

    private static final double targetVariance = 0.95;
    private static final double samplingPercentage = 0.0003;
    private static final Logger log = LogManager.getLogger(EnsembleQueryHandler_unnecessary.class);
    private String filename = "driver.txt";

    public static void main(String arg[]) {

        Map<String, List> opMap = CountyClusters.extractCountyGroups("./src/main/java/org/sustain/handlers/clusters.csv");
        System.out.println(opMap);
    }

    public EnsembleQueryHandler_unnecessary(ModelRequest request, StreamObserver<ModelResponse> responseObserver, SparkManager sparkManager) {
        super(request, responseObserver, sparkManager);
    }

    protected class GBRegressionTask implements SparkTask<List<ModelResponse>> {
        private final GBoostRegressionRequest gbRequest;
        private final Collection requestCollection;
        private final List<String> gisJoins;
        private Map<String, String> reverseChildToParentMap = null;

        public GBRegressionTask(ModelRequest modelRequest, List<String> gisJoins, Map<String, String> reverseChildToParentMap) {
            this.gbRequest = modelRequest.getGBoostRegressionRequest();
            this.requestCollection = modelRequest.getCollections(0); // We only support 1 collection currently
            this.gisJoins = gisJoins;
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

                String parent = "none";

                if(reverseChildToParentMap.get(gisJoin) != null){
                    parent = reverseChildToParentMap.get(gisJoin);
                }
                SummaryCalculator model = new SummaryCalculator(mongoUri, dbName, collection.getName(), gisJoin);
                model.parentGisJoin = parent;
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


                // Submit task to Spark Manager
                boolean ok = model.train();

                if (ok) {

                    GBoostRegressionResponse rsp = GBoostRegressionResponse.newBuilder()
                            .setGisJoin(model.getGisJoin())
                            .build();

                    modelResponses.add(ModelResponse.newBuilder()
                            .setGBoostRegressionResponse(rsp)
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

        String full_log_string = "";
        if (isValid(this.request)) {

            // PARSE THE CLUSTER CSV
            Map<String, List> clusterCSVMap = CountyClusters.extractCountyGroups("./src/main/java/org/sustain/handlers/clusters.csv");
            full_log_string+=FancyLogger.fancy_logging("CLUSTER_MAP: "+clusterCSVMap, null);

            if(request.getType().equals(ModelType.G_BOOST_REGRESSION)) {

                try {
                    GBoostRegressionRequest req = this.request.getGBoostRegressionRequest();

                    Map<String,String> reverseChildToParentMap = new HashMap<>();

                    // IGNORE GISJOIN SENT BY THE REQUEST. INSTEAD GET THE FIRST ENTRY
                    java.util.Iterator<Map.Entry<String, List>> iterator = clusterCSVMap.entrySet().iterator();

                    // A COMBINED LIST OF ALL DEPENDENT GISJOINS
                    List<String> allGISList = new ArrayList<>();

                    while (iterator.hasNext()) {
                        Map.Entry<String, List> firstEntry = iterator.next();
                        String parentGisJoin = firstEntry.getKey();
                        List<String> childrenGisJoin = firstEntry.getValue();
                        allGISList.addAll(childrenGisJoin);
                        allGISList.add(parentGisJoin);

                        for(String child: childrenGisJoin) {
                            reverseChildToParentMap.put(child, parentGisJoin);
                        }
                    }

                    List<List<String>> gisJoinBatches = batchGisJoins(allGISList, 20);

                    // ****************START PARENT TRAINING ***********************
                    List<Future<List<ModelResponse>>> batchedSummaryTasks = new ArrayList<>();
                    for (List<String> gisJoinBatch: gisJoinBatches) {
                        GBRegressionTask gbTask = new GBRegressionTask(this.request, gisJoinBatch, reverseChildToParentMap);
                        batchedSummaryTasks.add(this.sparkManager.submit(gbTask, "gb-regression-query"));
                    }

                    // Wait for each task to complete and return their ModelResponses
                    for (Future<List<ModelResponse>> indvTask: batchedSummaryTasks) {
                        List<ModelResponse> batchedModelResponses = indvTask.get();
                        for (ModelResponse modelResponse: batchedModelResponses) {
                            //DON'T THINK WE NEED RESPONSE OBSERVER HERE....NOTHING TO PASS BACK. JUST OBSERVE THE RESULTS
                            full_log_string+=FancyLogger.fancy_logging("RECEIVED A RESPONSE FOR "+modelResponse.getGBoostRegressionResponse().getGisJoin(),log);
                            this.responseObserver.onNext(modelResponse);

                        }
                    }

                    System.out.println(full_log_string);
                    FancyLogger.write_out(full_log_string, filename);



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
