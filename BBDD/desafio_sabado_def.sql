-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema desafio
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema desafio
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `desafio` DEFAULT CHARACTER SET utf8 ;
USE `desafio` ;

-- -----------------------------------------------------
-- Table `desafio`.`agent`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `desafio`.`agent` (
  `agent_id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NULL,
  `surname` VARCHAR(45) NULL,
  `email` VARCHAR(45) NOT NULL,
  `password` VARCHAR(100) NOT NULL,
  `telephone` VARCHAR(25) NULL,
  PRIMARY KEY (`agent_id`),
  UNIQUE INDEX `email_UNIQUE` (`email` ASC),
  UNIQUE INDEX `telephone_UNIQUE` (`telephone` ASC))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `desafio`.`silver`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `desafio`.`silver` (
  `silver_id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `surname` VARCHAR(45) NOT NULL,
  `address` VARCHAR(45) NOT NULL,
  `city` VARCHAR(45) NULL,
  `postal_code` INT NOT NULL,
  `dni_nie` VARCHAR(45) NOT NULL,
  `birthday` DATE NULL,
  `gender` VARCHAR(15) NULL,
  `marital_status` VARCHAR(45) NULL,
  `telephone` VARCHAR(45) NULL,
  `email` VARCHAR(45) NULL,
  `social_security_number` VARCHAR(45) NULL,
  `agent_id` INT NOT NULL,
  `results` VARCHAR(200) NULL,
  `new_valuation_date` DATE NULL,
  `contact_person` VARCHAR(45) NULL,
  `contact_p_relation` VARCHAR(45) NULL,
  `contact_p_telephone` VARCHAR(45) NULL,
  PRIMARY KEY (`silver_id`),
  INDEX `fk_silver_agent_idx` (`agent_id` ASC),
  CONSTRAINT `fk_silver_agent`
    FOREIGN KEY (`agent_id`)
    REFERENCES `desafio`.`agent` (`agent_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `desafio`.`dependency_form`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `desafio`.`dependency_form` (
  `date` DATE NOT NULL,
  `feeding` INT NOT NULL,
  `bathing` INT NOT NULL,
  `grooming` INT NOT NULL,
  `dressing` INT NOT NULL,
  `bowels` INT NOT NULL,
  `bladder` INT NOT NULL,
  `toilet_use` INT NOT NULL,
  `transfers` INT NOT NULL,
  `mobility` INT NOT NULL,
  `stairs` INT NOT NULL,
  `sum` INT NOT NULL,
  `enviar` BINARY(2) NOT NULL,
  `dep_form_id` INT NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`dep_form_id`),
  UNIQUE INDEX `dep_form_id_UNIQUE` (`dep_form_id` ASC))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `desafio`.`loneliness_form`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `desafio`.`loneliness_form` (
  `lon_form_id` INT NOT NULL AUTO_INCREMENT,
  `date` DATE NOT NULL,
  `q1` INT NOT NULL,
  `q2` INT NOT NULL,
  `q3` INT NOT NULL,
  `q4` INT NOT NULL,
  `q5` INT NOT NULL,
  `q6` INT NOT NULL,
  `q7` INT NOT NULL,
  `q8` INT NOT NULL,
  `q9` INT NOT NULL,
  `q10` INT NOT NULL,
  `sum` INT NOT NULL,
  `enviar` TINYINT(1) NULL,
  `silver_id` INT NOT NULL,
  `observations` VARCHAR(1000) NULL,
  INDEX `fk_loneliness_form_silver1_idx` (`silver_id` ASC),
  PRIMARY KEY (`lon_form_id`),
  UNIQUE INDEX `form_id_UNIQUE` (`lon_form_id` ASC),
  CONSTRAINT `fk_loneliness_form_silver1`
    FOREIGN KEY (`silver_id`)
    REFERENCES `desafio`.`silver` (`silver_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `desafio`.`resources`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `desafio`.`resources` (
  `resources_id` INT NOT NULL AUTO_INCREMENT,
  `day_care_center` TINYINT(1) NULL,
  `cofee_n_chat` TINYINT(1) NULL,
  `walking_club` TINYINT(1) NULL,
  `reading_club` TINYINT(1) NULL,
  `home_assistance` TINYINT(1) NULL,
  `phone_assistance` TINYINT(1) NULL,
  `garden_group` TINYINT(1) NULL,
  `cooking_group` TINYINT(1) NULL,
  `cycling_group` TINYINT(1) NULL,
  `board_games` TINYINT(1) NULL,
  `movie_club` TINYINT(1) NULL,
  `lon_form_id` INT NOT NULL,
  PRIMARY KEY (`resources_id`),
  INDEX `fk_resources_loneliness_form1_idx` (`lon_form_id` ASC),
  CONSTRAINT `fk_resources_loneliness_form1`
    FOREIGN KEY (`lon_form_id`)
    REFERENCES `desafio`.`loneliness_form` (`lon_form_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
