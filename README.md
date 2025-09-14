
** Description

Project to auto generate metadata given a set of data source in a specific format. 

** Requirements
- Input is either a set of data tables, parquet/csv file, etc. or a set of database tables.
- Output is a metadata file with specific fields, the format will be provided.
- The code should determine:
  - whether there is a primary key and/or candidate column(s) for primary. This will be done 
  checking the uniquness of the values in the data using candidate column(s). The candatates initially can be selected 
  from columns that has "id-looking" values, column name can give a clue, datatype can only be integer or string.
  - whether there is foreign keys in the data. This can be performed looking at all the sources in the dataset and their proposed 
  primary keys.
  - Type of the source as Entity, Event, State, Relationship or Participation.
    - Entity: The file/table contains attributes that belong to an entity. The primary key should only be id
    column that belongs to an entity. Ex: User file, Company file, etc. 
    - Event: events are things that happen at a specific point in time so event files/sources need to have a timestamp and attributes 
    that can be associated with an event. Ex: Transaction file. 
    - State: states are things that have lingering effect but observed at certain times so event files need to have a timestamp and 
    attributes that can be associated with states. Ex: Weather, macroeconomy, user marrital status, etc.
    - Relationship: The tables/files that contain 2 sets of entity ids as foreign keys.
    - Participation: The tables/files that contain an event id and an entity id as foreign keys.
