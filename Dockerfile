FROM maven:3-amazoncorretto-8 as build

# Switch to java dir to copy specific files
WORKDIR /usr/src/myapp/java
COPY java/data/ReinforcementLearning.xml java/data/ReinforcementLearningConfig.xml data/
COPY java/enemy_agents enemy_agents/
COPY java/src src/
COPY java/lib lib/
COPY java/pom.xml .

# Compile java
RUN mvn compile && \
    # Needed because I don't know if maven can have dependencies on .class so move it into the classpath
    mv enemy_agents/CombatAgentEnemy.class target/classes

# Copy python code
WORKDIR /usr/src/myapp/python
COPY python .

# Install python for our NN agent code
# The maven image uses redhat so we use yum not apt to install
# Apparently ask for a specific version either (=3.6.9)
# Also clean cache
RUN yum -y install python3 && \
    rm -rf /var/cache
RUN pip3 install --user --upgrade pip && \
    pip3 install --user -r requirements.txt

# Run java in background, then run python with it
# Need to cd because programs expect to be run from their directory
WORKDIR /usr/src/myapp
ENTRYPOINT cd java && mvn -f pom.xml exec:java -Dexec.mainClass="edu.cwru.sepia.Main2" -Dexec.classpathScope=compile \
           -Dexec.args="data/ReinforcementLearningConfig.xml" >> rl_learning_java_stdout.txt & \
           cd python && ./rl_training_enemy_agent.py >> rl_training_enemy_agent_stdout.txt

# Plan to copy only executable files. Doesn't seem necessary right now but might reduce image size?
# FROM openjdk:8
# COPY --from=build /usr/src/myapp/java/target/mavenGrpcTest-1.0-SNAPSHOT.jar /usr/src/myapp/java/mavenGrpcTest-1.0-SNAPSHOT.jar
# WORKDIR /usr/src/myapp/java
