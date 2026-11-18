package local.yanxi.backend.tasks;

import java.net.http.HttpClient;
import java.time.Duration;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.server.ResponseStatusException;

@Service
public class TaskQueryService {
    static final String TOKEN_HEADER = "X-Yanxi-Task-Token";
    private final RestClient python;

    public TaskQueryService(RestClient.Builder builder, @Value("${yanxi.python.base-url}") String baseUrl) {
        var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
        var requests = new JdkClientHttpRequestFactory(client);
        requests.setReadTimeout(Duration.ofSeconds(5));
        this.python = builder.baseUrl(baseUrl).requestFactory(requests).build();
    }

    public TaskList findTasks(String conversation, String token) {
        try {
            TaskList result = python.get()
                    .uri("/api/integration/tasks?conversation={conversation}", conversation)
                    .header(TOKEN_HEADER, token)
                    .retrieve()
                    .onStatus(status -> status.value() == 403, (request, response) -> {
                        throw new ResponseStatusException(HttpStatus.FORBIDDEN, "任务查询凭据无效。");
                    })
                    .onStatus(status -> status.value() == 404, (request, response) -> {
                        throw new ResponseStatusException(HttpStatus.NOT_FOUND, "对话不存在。");
                    })
                    .onStatus(status -> status.value() == 503, (request, response) -> {
                        throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Python 任务接口未启用。");
                    })
                    .body(TaskList.class);
            // 上游必须明确返回请求所指的对话，即使该对话暂时没有任务。
            if (result == null || !conversation.equals(result.conversation()) || result.tasks() == null) {
                throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Python 返回的任务数据不符合接口约定。");
            }
            return result;
        } catch (ResourceAccessException error) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Python 服务不可访问或响应超时。", error);
        } catch (RestClientException error) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Python 任务查询失败。", error);
        }
    }
}
