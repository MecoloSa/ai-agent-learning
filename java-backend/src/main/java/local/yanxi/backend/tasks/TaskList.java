package local.yanxi.backend.tasks;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public record TaskList(String conversation, long version, List<TaskSummary> tasks) {
    public record TaskSummary(
            @JsonProperty("task_id") String taskId,
            String title,
            @JsonProperty("activity_status") String activityStatus,
            @JsonProperty("understanding_status") String understandingStatus,
            @JsonProperty("is_cancelled") boolean cancelled
    ) {
    }
}
