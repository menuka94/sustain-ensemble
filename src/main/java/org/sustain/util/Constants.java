package org.sustain.util;

public class Constants {
    public static final String GIS_JOIN = "GISJOIN";

    public static class K8s {
        public static final String SPARK_K8S_MASTER = System.getenv("SPARK_K8S_MASTER");
        public static final boolean USE_KUBERNETES = Boolean.parseBoolean(System.getenv("USE_KUBERNETES"));
        public static final String SPARK_DOCKER_IMAGE = System.getenv("SPARK_DOCKER_IMAGE");
    }

    public static class Server {
        public static final String  HOST = System.getenv("SERVER_HOST");
        public static final Integer PORT = Integer.parseInt(System.getenv("SERVER_PORT"));
    }

    public static class DB {
        public static final String  NAME     = System.getenv("DB_NAME");
        public static final String  USERNAME = System.getenv("DB_USERNAME");
        public static final String  PASSWORD = System.getenv("DB_PASSWORD");
        public static final String  HOST     = System.getenv("DB_HOST");
        public static final Integer PORT     = Integer.parseInt(System.getenv("DB_PORT"));
    }

    public static class Spark {
        public static final String MASTER = System.getenv("SPARK_MASTER");
        public static final String EXECUTOR_CORES = System.getenv("SPARK_EXECUTOR_CORES");
        public static final String EXECUTOR_MEMORY = System.getenv("SPARK_EXECUTOR_MEMORY");
        public static final String INITIAL_EXECUTORS = System.getenv("SPARK_INITIAL_EXECUTORS");
        public static final String MIN_EXECUTORS = System.getenv("SPARK_MIN_EXECUTORS");
        public static final String MAX_EXECUTORS = System.getenv("SPARK_MAX_EXECUTORS");
        public static final String BACKLOG_TIMEOUT = System.getenv("SPARK_BACKLOG_TIMEOUT");
        public static final String IDLE_TIMEOUT = System.getenv("SPARK_IDLE_TIMEOUT");
    }
}
