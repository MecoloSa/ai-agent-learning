package local.yanxi.backend.tasks;

import java.util.regex.Pattern;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@RestController
public class TaskController {
    private static final Pattern CONVERSATION = Pattern.compile("original|[0-9a-f]{32}");
    private final TaskQueryService taskQuery;

    public TaskController(TaskQueryService taskQuery) {
        this.taskQuery = taskQuery;
    }

    @GetMapping("/api/conversations/{conversation}/tasks")
    public TaskList listTasks(
            @PathVariable("conversation") String conversation,
            @RequestHeader(name = TaskQueryService.TOKEN_HEADER, required = false) String token
    ) {
        if (!CONVERSATION.matcher(conversation).matches()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "无效的对话 ID。");
        }
        if (token == null || token.isBlank()) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "请提供任务查询凭据。");
        }
        return taskQuery.findTasks(conversation, token);
    }
}
