package local.yanxi.backend.tasks;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class TaskApiTest {
    private static final String SECOND = "a".repeat(32);
    private static final String MISMATCH = "b".repeat(32);
    private static final String HEADER = "X-Yanxi-Task-Token";
    private static final HttpServer UPSTREAM = startUpstream();

    @Autowired
    private MockMvc mvc;

    @DynamicPropertySource
    static void upstreamAddress(DynamicPropertyRegistry registry) {
        registry.add("yanxi.python.base-url", () -> "http://127.0.0.1:" + UPSTREAM.getAddress().getPort());
    }

    @AfterAll
    static void stopUpstream() {
        UPSTREAM.stop(0);
    }

    @Test
    void queriesOnlyTheRequestedConversation() throws Exception {
        mvc.perform(get("/api/conversations/{conversation}/tasks", "original").header(HEADER, "test-token"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.conversation").value("original"))
                .andExpect(jsonPath("$.tasks.length()").value(1))
                .andExpect(jsonPath("$.tasks[0].title").value("阅读工具定义"));
        mvc.perform(get("/api/conversations/{conversation}/tasks", SECOND).header(HEADER, "test-token"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.conversation").value(SECOND))
                .andExpect(jsonPath("$.tasks.length()").value(1))
                .andExpect(jsonPath("$.tasks[0].title").value("第二个对话的任务"));
    }

    @Test
    void keepsPermissionErrorsAndUnknownConversationsDistinct() throws Exception {
        mvc.perform(get("/api/conversations/original/tasks")).andExpect(status().isForbidden());
        mvc.perform(get("/api/conversations/original/tasks").header(HEADER, "wrong"))
                .andExpect(status().isForbidden());
        mvc.perform(get("/api/conversations/{conversation}/tasks", "f".repeat(32)).header(HEADER, "test-token"))
                .andExpect(status().isNotFound());
        mvc.perform(get("/api/conversations/invalid/tasks").header(HEADER, "test-token"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void rejectsAResponseForAnotherConversation() throws Exception {
        mvc.perform(get("/api/conversations/{conversation}/tasks", MISMATCH).header(HEADER, "test-token"))
                .andExpect(status().isBadGateway());
    }

    private static HttpServer startUpstream() {
        try {
            HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/api/integration/tasks", exchange -> {
                try (exchange) {
                    String query = exchange.getRequestURI().getQuery();
                    int status = 200;
                    String body;
                    if (!"test-token".equals(exchange.getRequestHeaders().getFirst(HEADER))) {
                        status = 403;
                        body = "{}";
                    } else if ("conversation=original".equals(query) || ("conversation=" + MISMATCH).equals(query)) {
                        body = taskJson("original", "阅读工具定义");
                    } else if (("conversation=" + SECOND).equals(query)) {
                        body = taskJson(SECOND, "第二个对话的任务");
                    } else {
                        status = 404;
                        body = "{}";
                    }
                    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
                    exchange.getResponseHeaders().set("Content-Type", "application/json");
                    exchange.sendResponseHeaders(status, bytes.length);
                    exchange.getResponseBody().write(bytes);
                }
            });
            server.start();
            return server;
        } catch (IOException error) {
            throw new IllegalStateException("无法启动任务接口测试替身", error);
        }
    }

    private static String taskJson(String conversation, String title) {
        return """
                {"conversation":"%s","version":2,"tasks":[
                  {"task_id":"read-001","title":"%s","activity_status":"not_started",
                   "understanding_status":"unknown","is_cancelled":false}
                ]}
                """.formatted(conversation, title);
    }
}
