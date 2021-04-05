package agents.interfaces;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;

import java.util.List;
import java.util.Map;

public interface AgentInterface {
    int[] observeUnitState(Integer unitID, State.StateView state,
                           List<Integer> myUnitIDs, List<Integer> enemyUnitIDs);

    float getUnitLastReward(Integer unitID, State.StateView state, History.HistoryView history,
                            List<Integer> myUnitIDs, List<Integer> enemyUnitIDs, boolean isFinalStep);

    void requestUnitAction(Integer unitID, int action, Map<Integer, Action> actions,
                           State.StateView state, List<Integer> enemyUnitIDs);
}
