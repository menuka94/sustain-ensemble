/* ========================================================
 * GBoostRegressionModel.java -
 *      Defines a generalized gradient boost regression model that can be
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
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.sustain.util.Constants;
import org.sustain.util.FancyLogger;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.spark.sql.functions.*;

// PERFORMS EXHAUSTIVE TRAINING OVER A SET OF PARAMETERS

/**
 * Provides an interface for building generalized Gradient Boost Regression
 * models on data pulled in using Mongo's Spark Connector.
 */
public class SummaryCalculator {

    private String filename="";

    private Dataset<Row> mongoCollection;
    // DATABASE PARAMETERS
    protected static final Logger log = LogManager.getLogger(SummaryCalculator.class);
    private String database, collection, mongoUri;
    private String[] features;
    private String label, gisJoin;

    public String parentGisJoin = "";
    private double parent_rmse = 0.0;

    String queryField = "gis_join";


    public Dataset<Row> getMongoCollection() {
        return mongoCollection;
    }

    public void setMongoCollection(Dataset<Row> mongoCollection) {
        this.mongoCollection = mongoCollection;
    }


    public SummaryCalculator(String mongoUri, String database, String collection, String gisJoin) {
        log.info("Gradient Boosting constructor invoked");
        setMongoUri(mongoUri);
        setDatabase(database);
        setCollection(collection);
        setGisjoin(gisJoin);
        filename = "summary/"+parentGisJoin+"_"+gisJoin+".txt";
    }

    public void setFilename() {
        this.filename = "summary/"+gisJoin+"_"+parentGisJoin+".txt";
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

        String msg = "";
        msg = "Initiating Gradient Boost Modelling...";

        FancyLogger.write_out(fancy_logging(msg), filename);

        // Select just the columns we want, discard the rest
        Dataset<Row> selected = mongoCollection.select("_id", desiredColumns());

        msg = "Data Fetch Completed in "+ calc_interval(startTime)+" secs";

        FancyLogger.write_out(fancy_logging(msg), filename);

        // PREPARING DATASET2
        Dataset<Row> gisDataset2 = selected.filter(selected.col(queryField).equalTo(gisJoin));

        List<Row> rows = gisDataset2.agg(stddev("relative_humidity_percent"), avg("relative_humidity_percent")).collectAsList();

        String fullSummary = FancyLogger.fancy_logging(rows+"") + "\n";
        FancyLogger.write_out(fullSummary,filename);


        return true;
    }



}