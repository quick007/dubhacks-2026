import {
  Bus,
  Car,
  FlaskConical,
  GraduationCap,
  Mountain,
  ShoppingBag,
  Tag,
  type LucideIcon,
} from "lucide-react";

export interface ProfileFact {
  icon: LucideIcon;
  fact: string;
}

export interface Profile {
  age: number;
  facts: ProfileFact[];
}

/**
 * What a large store would already know about each opening shopper. Illustrative and synthetic.
 * The frozen model run reads only the request.
 */
export const profiles: Record<string, Profile> = {
  "dev-sofa": {
    age: 20,
    facts: [
      { icon: GraduationCap, fact: "UW junior, CS major" },
      { icon: Bus, fact: "Takes the bus" },
      { icon: Tag, fact: "Sorts by lowest price" },
      { icon: ShoppingBag, fact: "Viewed space heaters" },
    ],
  },
  "ho-lunch": {
    age: 34,
    facts: [
      { icon: FlaskConical, fact: "Lab scientist" },
      { icon: Car, fact: "Drives to work" },
      { icon: Mountain, fact: "Hikes on weekends" },
      { icon: ShoppingBag, fact: "Bought a steel bottle" },
    ],
  },
};
