/* ========================================================
 * RFRegressionModel.java -
 *      Defines a generalized random forest regression model that can be
 *      built and executed over a set of MongoDB documents.
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
package org.sustain.modeling;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.sustain.util.Constants;
import org.sustain.util.FancyLogger;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Provides an interface for building generalized Random Forest Regression
 * models on data pulled in using Mongo's Spark Connector.
 */
public class RFChildIncrementalModel {

    private String filename="";

    private Dataset<Row> mongoCollection;

    // DATABASE PARAMETERS
    protected static final Logger log = LogManager.getLogger(RFChildIncrementalModel.class);
    private String database, collection, mongoUri;
    private String[] features;
    private String label, gisJoin;

    // MODEL PARAMETERS
    //Minimum information gain for a split to be considered at a tree node. default 0.0
    private Double minInfoGain = null;
    // Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Must be at least 1. (default = 1)
    private Integer minInstancesPerNode = null;
    //Minimum fraction of the weighted sample count that each child must have after split. Should be in the interval [0.0, 0.5). (default = 0.0)
    private Double minWeightFractionPerNode = null;
    //Whether bootstrap samples are used when building trees.
    private Boolean isBootstrap = null;
    //Fraction of the training data used for learning each decision tree, in range (0, 1]. (default = 1.0)
    private Double subsamplingRate = null;
    //Number of trees to train (at least 1). If 1, then no bootstrapping is used. If greater than 1, then bootstrapping is done.
    private Integer numTrees = 1;
    // Number of features to consider for splits at each node. Supported: "auto", "all", "sqrt", "log2", "onethird".
    // If "auto" is set, this parameter is set based on numTrees: if numTrees == 1, set to "all"; if numTrees > 1 (forest) set to "onethird".
    private String featureSubsetStrategy = null; //auto/all/sqrt/log2/onethird
    //Criterion used for information gain calculation. Supported values: "variance".
    private String impurity = null;
    //maxDepth - Maximum depth of the tree. (e.g., depth 0 means 1 leaf node, depth 1 means 1 internal node + 2 leaf nodes). (suggested value: 4)
    private Integer maxDepth = null;
    //maxBins - Maximum number of bins used for splitting features. (suggested value: 100)
    private Integer maxBins = null;
    private Double trainSplit = 0.8d;
    String errorType = "rmse";
    String queryField = "gis_join";

    private RandomForestRegressionModel parent_rfModel;
    public String parentGisJoin = "";
    private double parent_rmse = 0.0;

    public RandomForestRegressionModel getParent_rfModel() {
        return parent_rfModel;
    }

    public void setParent_rfModel(RandomForestRegressionModel parent_rfModel) {
        this.parent_rfModel = parent_rfModel;
    }

    public String getParentGisJoin() {
        return parentGisJoin;
    }

    public void setParentGisJoin(String parentGisJoin) {
        this.parentGisJoin = parentGisJoin;
    }

    public double getParent_rmse() {
        return parent_rmse;
    }

    public void setParent_rmse(double parent_rmse) {
        this.parent_rmse = parent_rmse;
    }

    double rmse = 0.0;
    private double r2 = 0.0;


    public Double getMinInfoGain() {
        return minInfoGain;
    }

    public void setMinInfoGain(Double minInfoGain) {
        this.minInfoGain = minInfoGain;
    }

    public Integer getMinInstancesPerNode() {
        return minInstancesPerNode;
    }

    public void setMinInstancesPerNode(Integer minInstancesPerNode) {
        this.minInstancesPerNode = minInstancesPerNode;
    }

    public Double getMinWeightFractionPerNode() {
        return minWeightFractionPerNode;
    }

    public void setMinWeightFractionPerNode(Double minWeightFractionPerNode) {
        this.minWeightFractionPerNode = minWeightFractionPerNode;
    }

    public double getR2() {
        return r2;
    }

    public void setR2(double r2) {
        this.r2 = r2;
    }

    public double getRmse() {
        return rmse;
    }

    public void setRmse(double rmse) {
        this.rmse = rmse;
    }

    public void setTrainSplit(Double trainSplit) {
        this.trainSplit = trainSplit;
    }

    public RFChildIncrementalModel(String mongoUri, String database, String collection, String gisJoin) {
        log.info("Random Forest constructor invoked");
        setMongoUri(mongoUri);
        setDatabase(database);
        setCollection(collection);
        setGisjoin(gisJoin);
        filename = "children/"+parentGisJoin+"_"+gisJoin+".txt";
    }

    public void setFilename() {
        this.filename = "children/"+parentGisJoin+"_"+gisJoin+".txt";
    }

    public Dataset<Row> getMongoCollection() {
        return mongoCollection;
    }

    public void setMongoCollection(Dataset<Row> mongoCollection) {
        this.mongoCollection = mongoCollection;
    }

    public String getDatabase() {
        return database;
    }

    public void setDatabase(String database) {
        this.database = database;
    }

    public String getCollection() {
        return collection;
    }

    public void setCollection(String collection) {
        this.collection = collection;
    }

    public String getMongoUri() {
        return mongoUri;
    }

    public void setMongoUri(String mongoUri) {
        this.mongoUri = mongoUri;
    }

    public void setFeatures(String[] features) {
        this.features = features;
    }

    public void setGisjoin(String gisJoin) {
        this.gisJoin = gisJoin;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String[] getFeatures() {
        return features;
    }

    public String getGisJoin() {
        return gisJoin;
    }

    public String getLabel() {
        return label;
    }

    public Boolean getBootstrap() {
        return isBootstrap;
    }

    public void setBootstrap(Boolean bootstrap) {
        isBootstrap = bootstrap;
    }

    public Double getSubsamplingRate() {
        return subsamplingRate;
    }

    public void setSubsamplingRate(Double subsamplingRate) {
        this.subsamplingRate = subsamplingRate;
    }

    public Integer getNumTrees() {
        return numTrees;
    }

    public void setNumTrees(Integer numTrees) {
        this.numTrees = numTrees;
    }

    public String getFeatureSubsetStrategy() {
        return featureSubsetStrategy;
    }

    public void setFeatureSubsetStrategy(String featureSubsetStrategy) {
        this.featureSubsetStrategy = featureSubsetStrategy;
    }

    public String getImpurity() {
        return impurity;
    }

    public void setImpurity(String impurity) {
        this.impurity = impurity;
    }

    public Integer getMaxDepth() {
        return maxDepth;
    }

    public void setMaxDepth(Integer maxDepth) {
        this.maxDepth = maxDepth;
    }

    public Integer getMaxBins() {
        return maxBins;
    }

    public void setMaxBins(Integer maxBins) {
        this.maxBins = maxBins;
    }

    private Seq<String> desiredColumns() {
        List<String> cols = new ArrayList<>();
        cols.add(queryField);
        Collections.addAll(cols, this.features);
        cols.add(this.label);
        return convertListToSeq(cols);
    }

    /**
     * Converts a Java List<String> of inputs to a Scala Seq<String>
     * @param inputList The Java List<String> we wish to transform
     * @return A Scala Seq<String> representing the original input list
     */
    public Seq<String> convertListToSeq(List<String> inputList) {
        return JavaConverters.asScalaIteratorConverter(inputList.iterator()).asScala().toSeq();
    }

    private String fancy_logging(String msg){

        String logStr = "\n============================================================================================================\n";
        logStr+=msg;
        logStr+="\n============================================================================================================";

        log.info(logStr);
        return logStr;
    }

    private double calc_interval(double startTime) {
        return ((double)System.currentTimeMillis() - startTime)/1000;
    }

    /**
     * Creates Spark context and trains the distributed model
     */
    public Boolean train() {

        //addClusterDependencyJars(sparkContext);
        double startTime = System.currentTimeMillis();

        FancyLogger.write_out(fancy_logging("Initiating Random Forest Modelling..."), filename);

        // Select just the columns we want, discard the rest
        Dataset<Row> selected = mongoCollection.select("_id", desiredColumns());

        FancyLogger.write_out(fancy_logging("Data Fetch Completed in "+ calc_interval(startTime)+" secs"), filename);
        startTime = System.currentTimeMillis();

        Dataset<Row> gisDataset = selected.filter(selected.col(queryField).equalTo(gisJoin))
                .withColumnRenamed(this.label, "label"); // Rename the chosen label column to "label"

        log.info("DATA TYPES: \n"+Arrays.toString(gisDataset.columns())+" "+gisDataset.dtypes());

        // Create a VectorAssembler to assemble all the feature columns into a single column vector named "features"
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(this.features)
                .setOutputCol("features");

        /* ITERATIVE SAMPLING OF THE mergedDataset*/
        float trainFraction = 0.3f;
        // PREPARING DATASET2
        Dataset<Row> gisDataset2 = selected.filter(selected.col(queryField).equalTo(gisJoin))
                .withColumnRenamed(this.label, "label"); // Rename the chosen label column to "label"

        //gisDataset2 = gisDataset2.sample(0.1);

        Dataset<Row> mergedDataset_transfer = vectorAssembler.transform(gisDataset2).cache();

        // COPYING PARAMETERS FROM PRE_TRAINED MODEL
        RandomForestRegressionModel rf2_model = parent_rfModel.copy(new ParamMap());
        RandomForestRegressor rf2 = (RandomForestRegressor) rf2_model.parent();

        int iter = 0;
        double targetRMSE = getParent_rmse();
        String fullSummary = "";

        long startTime_overall = System.currentTimeMillis();

        boolean converged = true;
        while (true) {
            startTime = System.currentTimeMillis();

            Dataset<Row> workingDataset = mergedDataset_transfer.sample(trainFraction);
            Dataset<Row>[] rds_transfer = workingDataset.randomSplit(new double[]{trainSplit, 1.0d - trainSplit});
            Dataset<Row> trainrdd_transfer = rds_transfer[0].cache();
            Dataset<Row> testrdd_transfer = rds_transfer[1];
            fancy_logging("Model Data Split completed in " + calc_interval(startTime));

            /* TRAIN PHASE */

            RandomForestRegressionModel rf2Model_iter = rf2.fit(trainrdd_transfer);
            fullSummary += FancyLogger.fancy_logging("Model Training Round " + iter + " completed in " + calc_interval(startTime)) +"\n";

            /* TEST/EVALUATION PHASE */
            startTime = System.currentTimeMillis();
            Dataset<Row> pred_pair = rf2Model_iter.transform(testrdd_transfer).select("label", "prediction").cache();
            RegressionMetrics metrics = new RegressionMetrics(pred_pair);
            this.rmse = metrics.rootMeanSquaredError();

            fullSummary += FancyLogger.fancy_logging("Model Evaluation/Loss Computation Round " + iter + " completed in " + calc_interval(startTime)
                    + "\nEVALUATIONS: RMSE, R2: " + rmse + " " + r2) + "\n";

            logModelResults();

            trainFraction = trainFraction * 2;

            if (this.rmse < targetRMSE) {
                fullSummary += FancyLogger.fancy_logging("DESIRED ACCURACY ACHIEVED... EVALUATION TIME"+targetRMSE+" "+this.rmse) + "\n";
                converged = true;
                break;
            } else if (trainFraction > 0.9) {
                fullSummary += FancyLogger.fancy_logging("DESIRED ACCURACY NOT ACHIEVED... RAN OUT OF SAMPLES") + "\n";
                converged = false;
                break;
            } else {
                fullSummary += FancyLogger.fancy_logging("DESIRED ACCURACY NOT ACHIEVED " + targetRMSE+" "+this.rmse + " ...RETRAINING") + "\n";

            }

            // COPYING OVER PARAMETERS FROM PREVIOUS ITERATION
            RandomForestRegressor rf2_tmp = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("label");
            rf2_tmp.copy(rf2Model_iter.paramMap());
            rf2 = rf2_tmp;
            iter++;
        }

        if(converged) {
            fullSummary += FancyLogger.fancy_logging("OVERALL CONVERGENCE TIME " + calc_interval(startTime_overall)
                    + "\nEVALUATIONS: RMSE, R2: " + rmse + " " + r2) + "\n";
        } else {
            fullSummary += FancyLogger.fancy_logging("OVERALL CONVERGENCE TIME XXX " + calc_interval(startTime_overall)
                    + "\nEVALUATIONS: RMSE, R2: " + rmse + " " + r2) + "\n";
        }
        FancyLogger.write_out(fullSummary,filename);


        return true;
    }

    /**
     * Injecting user-defined parameters into model
     * @param rf - Random Forest Regression model Object
     */
    private void ingestParameters(RandomForestRegressor rf) {
        if (this.isBootstrap != null) {
            rf.setBootstrap(this.isBootstrap);
        }
        if (this.subsamplingRate != null) {
            rf.setSubsamplingRate(this.subsamplingRate);
        }
        if (this.numTrees != null) {
            rf.setNumTrees(this.numTrees);
        }
        if (this.featureSubsetStrategy != null) {
            rf.setFeatureSubsetStrategy(this.featureSubsetStrategy);
        }
        if (this.impurity != null) {
            rf.setImpurity(this.impurity);
        }
        if (this.maxDepth != null) {
            rf.setMaxDepth(this.maxDepth);
        }
        if (this.maxBins != null) {
            rf.setMaxBins(this.maxBins);
        }

        if (this.minInfoGain != null) {
            rf.setMinInfoGain(this.minInfoGain);
        }

        if (this.minInstancesPerNode != null) {
            rf.setMinInstancesPerNode(this.minInstancesPerNode);
        }

        if (this.minWeightFractionPerNode != null) {
            rf.setMinWeightFractionPerNode(this.minWeightFractionPerNode);
        }

    }

    private void addClusterDependencyJars(JavaSparkContext sparkContext) {
        String[] jarPaths = {
                "build/libs/mongo-spark-connector_2.12-3.0.1.jar",
                "build/libs/spark-core_2.12-3.0.1.jar",
                "build/libs/spark-mllib_2.12-3.0.1.jar",
                "build/libs/spark-sql_2.12-3.0.1.jar",
                "build/libs/bson-4.0.5.jar",
                "build/libs/mongo-java-driver-3.12.5.jar",
                //"build/libs/mongodb-driver-core-4.0.5.jar"
        };

        for (String jar: jarPaths) {
            log.info("Adding dependency JAR to the Spark Context: {}", jar);
            sparkContext.addJar(jar);
        }
    }

    public void populateTest() {
        this.numTrees = 1;
    }

    private void logModelResults() {
        log.info("Results for GISJoin {}\n" +
                        "RMSE: {}\n" +
                        "R2: {}\n"
                ,
                this.gisJoin, this.rmse, this.r2);
    }

    /**
     * Used exclusively for testing and running a linear model directly, without having to interface with gRPC.
     * @param args Usually not used.
     */
    public static void main(String[] args) {
        String[] features = {"max_eastward_wind","max_min_air_temperature"};
        String label = "min_eastward_wind";
        String gisJoins = "G0100290";
        String collection_name = "macav2";

        SparkSession sparkSession = SparkSession.builder()
                .master(Constants.Spark.MASTER)
                .appName("SUSTAIN RForest Regression Model")
                .config("spark.mongodb.input.uri", String.format("mongodb://%s:%d", Constants.DB.HOST, Constants.DB.PORT))
                .config("spark.mongodb.input.database", Constants.DB.NAME)
                .config("spark.mongodb.input.collection", "maca_v2")
                .getOrCreate();

        JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext());
        ReadConfig readConfig = ReadConfig.create(sparkContext);

        RFChildIncrementalModel rfModel = new RFChildIncrementalModel(
                "mongodb://lattice-46:27017", "sustaindb", collection_name, gisJoins);
        rfModel.setMongoCollection(MongoSpark.load(sparkContext, readConfig).toDF());
        rfModel.populateTest();
        rfModel.setFeatures(features);
        rfModel.setLabel(label);
        rfModel.setGisjoin(gisJoins);

        rfModel.train();
        log.info("Executed rfModel.main() successfully");
        sparkContext.close();
    }



}