
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

public class ConcTestRestful {
    private static int dimension = 300;
    private static String collectionName = "test_glove";
    private static String vectorFieldName = "vector";
    private static int topk = 100;
    private static List<byte[]> randomSearchContents = new ArrayList<>();
    private static List<Integer> threadCountsToTest = Arrays.asList(1, 10, 20, 40, 100, 150, 200, 300, 400, 500);
    private static int elapseTime = 300;
    private static int intermissionTime = 30;
    private static Map<String, Object> configurations;
    private static String searchUrl = "http://localhost:19530/v2/vectordb/entities/search";
    private static String token = "root:Milvus";

    private static final Random RANDOM = new Random();

    private static List<List<Float>> generateFloatVectors(int count) {
        List<List<Float>> vectors = new ArrayList<>();
        for (int n = 0; n < count; ++n) {
            List<Float> vector = new ArrayList<>();
            for (int i = 0; i < dimension; ++i) {
                vector.add(RANDOM.nextFloat());
            }
            vectors.add(vector);
        }

        return vectors;
    }

    private static void prepareRandomSearchContents(int count) {
        for (int i = 0; i < count; i++) {
            List<List<Float>> vectors = generateFloatVectors(1);
            String strContent = String.format(
                    "{\"collectionName\":\"%s\", \"annsField\": \"%s\", \"limit\": %d, \"consistency_level\": \"Bounded\", \"data\": %s, \"searchParams\": {\"params\": {\"level\": 1}}}",
                    collectionName, vectorFieldName, topk, vectors);
            byte[] byteContent = strContent.getBytes(StandardCharsets.UTF_8);
            randomSearchContents.add(byteContent);
        }
    }

    private static HttpURLConnection getHttpURLConnection(String url, byte[] body) throws Exception {
        URL obj = new URL(url);
        HttpURLConnection conn = (HttpURLConnection) obj.openConnection();
        conn.setDoOutput(true);
        conn.setDoInput(true);

        conn.setRequestMethod("POST");
        conn.setRequestProperty("Authorization", String.format("Bearer %s", token));
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");

        try (OutputStream os = conn.getOutputStream()) {
            os.write(body);
        }
        return conn;
    }

    private static void randomSearch() throws Exception {
        int idx = RANDOM.nextInt(randomSearchContents.size());
        byte[] body = randomSearchContents.get(idx);
        HttpURLConnection conn = getHttpURLConnection(searchUrl, body);
        sendRequest(conn);
    }

    private static void sendRequest(HttpURLConnection conn) throws Exception {
        try {
            int responseCode = conn.getResponseCode();
            // System.out.println("Response Code : " + responseCode);
            if (responseCode != 200) {
                throw new RuntimeException("Failed to search, error code: " + Integer.valueOf(responseCode));
            }

            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            String inputLine;
            StringBuffer response = new StringBuffer();
            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();
            // System.out.println(response);
        } catch (Exception e) {
            throw e;
        }
    }

    private static class SearchRunner implements Runnable {
        private final List<byte[]> searchContents;
        private long timeSpanMs;
        private long executedRequests = 0L;
        private List<Long> allLatencies = new ArrayList<>();
        private static final Random RANDOM = new Random();

        public SearchRunner(List<byte[]> searchContents, long timeSpanSeconds) {
            this.searchContents = searchContents;
            this.timeSpanMs = timeSpanSeconds * 1000;
        }

        public long executedReuests() {
            return this.executedRequests;
        }

        public long timeSpanSeconds() {
            return this.timeSpanMs;
        }

        public List<Long> allLatencies() {
            return this.allLatencies;
        }

        private void randomSearch() {
            int idx = RANDOM.nextInt(randomSearchContents.size());
            byte[] body = randomSearchContents.get(idx);
            try {
                long start = System.currentTimeMillis();
                sendRequest(getHttpURLConnection(searchUrl, body));
                allLatencies.add(System.currentTimeMillis() - start);
                executedRequests++;
            } catch (Exception e) {
                System.out.println(e);
            }
        }

        public void run() {
            System.out.println(String.format("Thread %s start to run %.3f seconds",
                    Thread.currentThread().getName(), this.timeSpanMs * 0.001));
            long start = System.currentTimeMillis();
            while (true) {
                randomSearch();
                if (System.currentTimeMillis() - start >= timeSpanMs) {
                    this.timeSpanMs = System.currentTimeMillis() - start;
                    System.out.println(String.format("Thread %s stop, %d requests finished in %.3f seconds",
                            Thread.currentThread().getName(), this.executedRequests, this.timeSpanMs * 0.001));
                    break;
                }
            }
        }
    }

    private static Map<String, String> parallelSearch(int threadCount, int timeSpan) {
        long tsStart = System.currentTimeMillis();
        List<Thread> threads = new ArrayList<>();
        List<SearchRunner> runners = new ArrayList<>();
        for (int i = 0; i < threadCount; i++) {
            SearchRunner runner = new SearchRunner(randomSearchContents, timeSpan);
            runners.add(runner);
            Thread t = new Thread(runner);
            threads.add(t);
            t.start();
        }

        try {
            for (Thread t : threads) {
                t.join();
            }
        } catch (InterruptedException e) {
            System.out.println("Thread interrupted");
            e.printStackTrace();
        }
        long tsEnd = System.currentTimeMillis();

        long totalExecutedRequests = 0L;
        long totalTimeSpanMs = 0L;
        List<Long> allLatencies = new ArrayList<>();
        for (SearchRunner runner : runners) {
            totalExecutedRequests += runner.executedReuests();
            totalTimeSpanMs += runner.timeSpanSeconds();
            allLatencies.addAll(runner.allLatencies());
        }

        float timeCostSec = (float) ((tsEnd - tsStart) * 0.001);
        float avgLatency = (float) totalTimeSpanMs / totalExecutedRequests;

        allLatencies.sort(Comparator.naturalOrder());
        int p99poz = (int) (allLatencies.size() * 0.99);
        long p99 = (p99poz >= allLatencies.size()) ? allLatencies.get(allLatencies.size() - 1)
                : allLatencies.get(p99poz);

        Map<String, String> info = new HashMap<>();
        info.put("ExecuteCount", String.format("%d requests executed", totalExecutedRequests));
        info.put("AvgLatency", String.format("Average latency: %.1f milliseconds", avgLatency));
        info.put("P99Latency", String.format("P99 latency: %d milliseconds", p99));
        info.put("QPS", String.format("QPS: %.1f", totalExecutedRequests / timeCostSec));
        return info;
    }

    private static void readConfig(String path) {
        Map<String, Object> configs = new HashMap<>();
        try (Stream<String> stream = Files.lines(Paths.get(path))) {
            stream.forEach((String line) -> {
                String[] pair = line.split("#");
                if (pair.length != 2) {
                    System.out.println("Invalid configuration: " + line);
                    throw new RuntimeException("Invalid configuration: " + line);
                }

                try {
                    if (pair[0].equals("uri") || pair[0].equals("token") || pair[0].equals("collection_name")
                            || pair[0].equals("vector_field")) {
                        configs.put(pair[0], pair[1]);
                    } else if (pair[0].equals("dim") || pair[0].equals("topk") || pair[0].equals("conc_duration")
                            || pair[0].equals("conc_intermission")) {
                        configs.put(pair[0], Integer.parseInt(pair[1]));
                    } else if (pair[0].equals("conc_list")) {
                        List<Integer> threadCounts = new ArrayList<>();
                        String[] values = pair[1].split(",");
                        for (String ss : values) {
                            if (ss.isEmpty()) {
                                continue;
                            }
                            threadCounts.add(Integer.parseInt(ss));
                        }
                        configs.put(pair[0], threadCounts);
                    }
                } catch (Exception e) {
                    throw e;
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(configs);
        configurations = configs;
    }

    private static void updateConfigurations() {
        if (configurations.containsKey("uri")) {
            searchUrl = configurations.get("uri") + "/v2/vectordb/entities/search";
        }
        if (configurations.containsKey("token")) {
            token = (String) configurations.get("token");
        }
        if (configurations.containsKey("collection_name")) {
            collectionName = (String) configurations.get("collection_name");
        }
        if (configurations.containsKey("vector_field")) {
            vectorFieldName = (String) configurations.get("vector_field");
        }
        if (configurations.containsKey("dim")) {
            dimension = (Integer) configurations.get("dim");
        }
        if (configurations.containsKey("topk")) {
            topk = (Integer) configurations.get("topk");
        }
        if (configurations.containsKey("conc_duration")) {
            elapseTime = (Integer) configurations.get("conc_duration");
        }
        if (configurations.containsKey("conc_intermission")) {
            intermissionTime = (Integer) configurations.get("conc_intermission");
        }
        if (configurations.containsKey("conc_list")) {
            threadCountsToTest = (List<Integer>) configurations.get("conc_list");
        }
    }

    public static void main(String[] args) {
        if (args.length > 0) {
            String currentPath = System.getProperty("user.dir");
            Path path = Paths.get(currentPath, args[0]);
            String fullPath = path.toString();
            System.out.println(fullPath);
            readConfig(fullPath);
            updateConfigurations();
        }

        int count = 10000;
        prepareRandomSearchContents(count);

        List<Map<String, String>> testResults = new ArrayList<>();
        for (Integer threadCount : threadCountsToTest) {
            Map<String, String> result = parallelSearch(threadCount, elapseTime);
            result.put("ThreadCount", String.format("Thread count: %d", threadCount));
            result.put("ElapseTime", String.format("Elapse time: %d seconds", elapseTime));
            testResults.add(result);

            try {
                System.out.println(String.format("Sleep intermission time %d seconds ... ", intermissionTime));
                Thread.sleep(intermissionTime * 1000L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.out.println("Thread was interrupted");
                return;
            }
        }

        for (Map<String, String> result : testResults) {
            System.out.println("\n=================================================");
            for (String ss : result.values()) {
                System.out.println(ss);
            }
        }
    }
}
