package agents.interfaces;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.util.Direction;

import java.util.List;
import java.util.Map;

public class QTableInterface implements AgentInterface {

    private int playernum;

    public QTableInterface(int playernum)
    {
        this.playernum = playernum;
    }

    /**
     * The relative location of another unit consists of 2 values:
     * the x and y components of the enemy
     * The agent sees in a 5x5 grid around itself with the top left being 0,0
     * If the enemy is too far away in one direction that direction gets a 5 + 1 = 6
     * while the other value is 2 (reserved value, because enemy would be overlapping unit)
     * which is impossible
     * @param unitID This unit
     * @param otherID The other unit
     * @param state Stateview
     * @return The x, y, or offscreen values
     */
    public int[] getUnitRelativeLocation(Integer unitID, Integer otherID, State.StateView state)
    {
        int range = 2; // How far away units can see

        Unit.UnitView unit = state.getUnit(unitID);
        Unit.UnitView enemy = state.getUnit(otherID);

        int dx = enemy.getXPosition() - unit.getXPosition();
        int dy = enemy.getYPosition() - unit.getYPosition();

        // Units expect to be player 1, in the left side of the field
        // Flip their state if they are on the other side of the field
        // so they get data they are used to
        if (playernum == 0)
        {
            dx *= -1;
            dy *= -1;
        }

        // If x or y is greater or less than range
        // clip the value and set the other to the reserved
        // value
        if (dx > range) {
            dx = range + 1;
            dy = 0;
        } else if (dx < -range) {
            dx = -range - 1;
            dy = 0;
        } else if (dy > range) {
            dy = range + 1;
            dx = 0;
        } else if (dy < -range) {
            dy = -range - 1;
            dx = 0;
        }

        // convert to 1 - 5
        // reserved value of 0 becomes 3 (in the middle)
        // 0 means "off to the left or down" 6 means "off to the right or up"
        dx += range + 1;
        dy += range + 1;

        return new int[] {dx, dy};
    }

    @Override
    public int[] observeUnitState(Integer unitID, State.StateView state, List<Integer> myUnitIDs, List<Integer> enemyUnitIDs)
    {
        // Indicates a dead unit
        int deadValue = 3;

        // 3 other units, x and y for each makes 6
        // A dead unit gets is placed at the location of
        // the current unit (which is impossible otherwise)
        // (that index is (3, 3) I believe)
        int[] environment = new int[6];

        // Get location of friendly unit
        // idx keeps track of where in state array we are
        int[] unitLoc;
        int idx = 0;
        for (Integer id : myUnitIDs)
        {
            // If not the other unit
            if (!id.equals(unitID))
            {
                unitLoc = getUnitRelativeLocation(unitID, id, state);
                environment[idx] = unitLoc[0];
                environment[idx + 1] = unitLoc[1];
                idx += 2;
            }
        }

        // If our friendly unit is dead
        // Put it in the dead state
        if (idx == 0)
        {
            environment[idx] = deadValue;
            environment[idx + 1] = deadValue;
            idx += 2;
        }

        // Fill in enemies' states
        // Or dead if they are dead
        for (Integer id : enemyUnitIDs)
        {
            unitLoc = getUnitRelativeLocation(unitID, id, state);
            environment[idx] = unitLoc[0];
            environment[idx + 1] = unitLoc[1];
            idx += 2;
        }

        // If any enemy units are dead
        // fill state in with the dead state
        while (idx < 6)
        {
            environment[idx] = deadValue;
            environment[idx + 1] = deadValue;
            idx += 2;
        }

        return environment;
    }

    @Override
    public float getUnitLastReward(Integer unitID, State.StateView state, History.HistoryView history,
                                   List<Integer> myUnitIDs, List<Integer> enemyUnitIDs, boolean isFinalStep) {

        if (isFinalStep)
        {
            return getUnitFinalLastReward(unitID, state, history, myUnitIDs, enemyUnitIDs);
        }

        float stepReward = -0.03f;
        float distanceReward = 0.009f;
        float damageReward = 0.05f;
        float enemyDamageReward = -0.03f;

        float deathReward = 1.0f;

        float reward = 0;

        // Punishment for taking too long
        reward += stepReward;

        int turnNum = state.getTurnNumber();

        // Incentivize getting close to enemy
        // Reward falls off as 1 / distance to enemy
        Unit.UnitView unit = state.getUnit(unitID);
        Unit.UnitView enemy = state.getUnit(enemyUnitIDs.get(0));

        int unitX = unit.getXPosition();
        int unitY = unit.getYPosition();
        int enemyX = enemy.getXPosition();
        int enemyY = enemy.getYPosition();

        reward += distanceReward / (Math.abs(enemyX - unitX) + Math.abs(enemyY - unitY));

        // Reward agent for attacking enemy
        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            int damage = damageLog.getDamage();

            // If we did the damage
            if (unitID.equals(damageLog.getAttackerID())) {
                reward += damageReward * damage; // Add to reward
            }
            else if (unitID.equals(damageLog.getDefenderID())) // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    public float getUnitFinalLastReward(Integer unitID, State.StateView state, History.HistoryView history,
                                     List<Integer> myUnitIDs, List<Integer> enemyUnitIDs)
    {
        float reward = 0;
        float stepReward = -0.03f;
        float damageReward = 0.05f;
        float enemyDamageReward = -0.03f;

        float winReward = 1.0f;
        float loseReward = 0.0f;

        // Punishment for taking too long
        reward += stepReward;

        // Reward for winning
        if (enemyUnitIDs.size() == 0)
        {
            reward += winReward;
        }

        if (myUnitIDs.size() == 0)
        {
            reward += loseReward;
        }

        int turnNum = state.getTurnNumber();

        // Reward agent for attacking enemy
        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            int damage = damageLog.getDamage();

            // If we did the damage
            if (unitID.equals(damageLog.getAttackerID())) {
                reward += damageReward * damage; // Add to reward
            }
            else if (unitID.equals(damageLog.getDefenderID())) // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    @Override
    public void requestUnitAction(Integer unitID, int action, Map<Integer, Action> actions,
                                  State.StateView state, List<Integer> enemyUnitIDs) {
        if (action >= 0 && action < 4) {
            // Agents expect to be player 1, on the left of the screen
            // When an agent on the right of the screen says to go right,
            // they actually mean left. This flips left/right, up/down for that agent
            if (playernum == 0) {
                action = (action + 2) % 4;
            }

            Direction dir = Direction.NORTH; // Default value so it compiles
            switch (action) {
                case 0:
                    dir = Direction.NORTH;
                    break;
                case 1:
                    dir = Direction.EAST;
                    break;
                case 2:
                    dir = Direction.SOUTH;
                    break;
                case 3:
                    dir = Direction.WEST;
                    break;
                default:
                    System.err.println("Error: Bad action " + action);
                    break;
            }
            actions.put(unitID, Action.createPrimitiveMove(unitID, dir));
        } else if (action == 4) {
            Unit.UnitView unit = state.getUnit(unitID);

            // Find the closest enemy and attack them
            // may not be in range
            Integer closestEnemyID = enemyUnitIDs.get(0);
            int closestEnemyDist = Integer.MAX_VALUE;
            for (Integer enemyID : enemyUnitIDs) {
                Unit.UnitView enemy = state.getUnit(enemyID);

                int dx = enemy.getXPosition() - unit.getXPosition();
                int dy = enemy.getYPosition() - unit.getYPosition();

                if (Math.max(dx, dy) < closestEnemyDist) {
                    closestEnemyDist = Math.max(dx, dy);
                    closestEnemyID = enemyID;
                }
            }

            actions.put(unitID, Action.createPrimitiveAttack(unitID, closestEnemyID));
        } else if (action == -2) {
            System.err.println("Error: Something went wrong, bad action " + action);
        } else {
            System.out.println("Note: noop action " + action);
        }
    }
}
