/*==============================================================*/
/* DBMS name:      MySQL 5.0                                    */
/* Created on:     2012/1/13 17:55:16                           */
/*==============================================================*/


drop index idx_app_domain on nap_apps;

drop index idx_app_guid on nap_apps;

drop table if exists nap_apps;

drop index idx_openid_user on nap_openids;

drop index idx_openid on nap_openids;

drop table if exists nap_openids;

drop table if exists nap_repositories;

drop table if exists nap_users;

drop table if exists nap_versions;

/*==============================================================*/
/* Table: nap_apps                                              */
/*==============================================================*/
create table nap_apps
(
   id                   int not null auto_increment,
   user                 int not null,
   site                 varchar(32) not null,
   name                 varchar(16) not null,
   outline              text,
   type                 tinyint not null,
   ico                  varchar(64),
   logo                 varchar(64),
   welcome              varchar(64),
   style                varchar(16),
   guid                 varchar(40) not null,
   domain               varchar(64) not null,
   home_url             varchar(64) not null,
   plugin_url           varchar(64) not null,
   create_time          timestamp not null default CURRENT_TIMESTAMP,
   primary key (id)
);

/*==============================================================*/
/* Index: idx_app_guid                                          */
/*==============================================================*/
create unique index idx_app_guid on nap_apps
(
   guid
);

/*==============================================================*/
/* Index: idx_app_domain                                        */
/*==============================================================*/
create unique index idx_app_domain on nap_apps
(
   domain
);

/*==============================================================*/
/* Table: nap_openids                                           */
/*==============================================================*/
create table nap_openids
(
   id                   int not null auto_increment,
   user                 int not null,
   name                 varchar(32),
   openid               varchar(64) not null,
   type                 tinyint not null,
   create_time          timestamp not null default CURRENT_TIMESTAMP,
   last_login           datetime,
   primary key (id)
);

/*==============================================================*/
/* Index: idx_openid                                            */
/*==============================================================*/
create unique index idx_openid on nap_openids
(
   openid,
   type
);

/*==============================================================*/
/* Index: idx_openid_user                                       */
/*==============================================================*/
create index idx_openid_user on nap_openids
(
   user
);

/*==============================================================*/
/* Table: nap_repositories                                      */
/*==============================================================*/
create table nap_repositories
(
   id                   int not null auto_increment,
   client_type          tinyint not null,
   version              varchar(16) not null,
   src                  varchar(64) not null,
   pub_date             date not null,
   create_time          timestamp not null default CURRENT_TIMESTAMP,
   changelog            text,
   primary key (id)
);

/*==============================================================*/
/* Table: nap_users                                             */
/*==============================================================*/
create table nap_users
(
   id                   int not null auto_increment,
   name                 varchar(32) not null,
   email                varchar(64),
   phone                varchar(16),
   validation           char(8) not null,
   status               tinyint not null,
   create_time          timestamp not null default CURRENT_TIMESTAMP,
   primary key (id)
);

/*==============================================================*/
/* Table: nap_versions                                          */
/*==============================================================*/
create table nap_versions
(
   id                   int not null auto_increment,
   app                  int not null,
   version              int not null,
   client_type          tinyint not null,
   app_path             varchar(64),
   dl_count             int not null,
   build_status         tinyint not null,
   create_time          timestamp not null default CURRENT_TIMESTAMP,
   build_begin_time     datetime,
   build_end_time       datetime,
   primary key (id)
);

alter table nap_apps add constraint fk_site_user foreign key (user)
      references nap_users (id);

alter table nap_openids add constraint fk_user_openid foreign key (user)
      references nap_users (id);

alter table nap_versions add constraint FK_fk_app_version foreign key (app)
      references nap_apps (id);

alter table nap_versions add constraint FK_fk_version_repository foreign key (version)
      references nap_repositories (id);

