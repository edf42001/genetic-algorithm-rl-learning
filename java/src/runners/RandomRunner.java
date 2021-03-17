/**
 *  Strategy Engine for Programming Intelligent Agents (SEPIA)
 Copyright (C) 2012 Case Western Reserve University
 This file is part of SEPIA.
 SEPIA is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 SEPIA is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with SEPIA.  If not, see <http://www.gnu.org/licenses/>.
 */
package runners;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.cwru.sepia.environment.model.SimpleModel;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.StateCreator;
import edu.cwru.sepia.environment.model.state.XmlStateCreator;
import edu.cwru.sepia.experiment.Configuration;
import edu.cwru.sepia.experiment.Runner;

import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.Environment;

/**
 * A {@code Runner} that runs a number of episodes using {@code edu.cwru.sepia.model.SimpleModel}.
 * @author Tim
 *
 */
public class RandomRunner extends Runner {
    private static final Logger logger = Logger.getLogger(RandomRunner.class.getCanonicalName());

    private int seed;
    private int numEpisodes;
    private int episodesPerSave;
    private boolean saveAgents;
    private Environment env;

    public RandomRunner(Configuration configuration, StateCreator stateCreator, Agent[] agents) {
        super(configuration, stateCreator, agents);
    }

    @Override
    public void run() {
        seed = configuration.getInt("experiment.RandomSeed", 6);
        numEpisodes = configuration.getInt("experiment.NumEpisodes", 1);
        episodesPerSave = configuration.getInt("experiment.EpisodesPerSave", 0);
        saveAgents = configuration.getBoolean("experiment.SaveAgents", false);

        stateCreator = new RandomStateCreator((XmlStateCreator) stateCreator);
        SimpleModel model = new SimpleModel(stateCreator.createState(), 0, stateCreator, configuration);

        env = new Environment(agents ,model, seed);
        for(int episode = 0; episode < numEpisodes; episode++) {
            try {
                env.runEpisode();
            } catch (InterruptedException e) {
                logger.log(Level.SEVERE, "Unable to complete episode " + episode, e);
            }
        }
    }

}